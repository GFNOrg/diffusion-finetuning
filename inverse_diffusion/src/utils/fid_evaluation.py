import math
import os
from itertools import combinations

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch import cosine_similarity
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm
from fld.metrics.FLD import FLD
from fld.metrics.FID import FID
from fld.metrics.AuthPct import AuthPct
from fld.metrics.CTTest import CTTest
from fld.metrics.KID import KID
from fld.metrics.PrecisionRecall import PrecisionRecall


class NoContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class DIVERSITY:
    def compute_metric(
        self,
        train_feat,
        test_feat,
        gen_feat,
        samples=5000,
        pairs=20000
    ):
        """
        Computes the average cosine similarity between all pairs of elements in a tensor.

        Args:
          features: A PyTorch tensor of shape (BS, N).
          max_pairs: (Optional) Maximum number of pairs to consider. Defaults to None (all pairs).

        Returns:
          A float representing the average cosine similarity.
        """
        batch_size, num_features = train_feat.shape

        samples = min(samples, batch_size)

        # Ensure max_pairs doesn't exceed total possible pairs
        sampled_indices = np.random.choice(list(range(batch_size)), samples, replace=False)
        idx = np.array(list(combinations(sampled_indices, 2)))
        pairs = min(len(idx), pairs)
        pairs_idx = np.random.choice(list(range(len(idx))), pairs, replace=False)
        idx = idx[pairs_idx]

        # Return the average cosine similarity
        return cosine_similarity(train_feat[idx[:, 0]], train_feat[idx[:, 1]], dim=1).mean()


class SCOREEvaluation:
    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        dl_test=None,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=50000,
        normalize_input=True,
        inception_block_idx=2048,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.dl_test = dl_test
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        if accelerator is not None:
            self.inception_v3 = accelerator.prepare(InceptionV3([block_idx], normalize_input=normalize_input)).to(device)
        else:
            self.inception_v3 = InceptionV3([block_idx], normalize_input=normalize_input).to(device)
        self.dataset_stats_loaded = False
        self.accelerator = accelerator
        accelerator.prepare(self.inception_v3)

        print(f"using {inception_block_idx} block for fid computation")

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    @torch.inference_mode()
    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.train_real_features, self.test_real_features = torch.FloatTensor(ckpt["train_real_features"]), torch.FloatTensor(ckpt["test_real_features"])
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            loaders = {'train': self.dl, 'test': self.dl_test}
            features = {"train_real_features": None, "test_real_features":None}
            for split in loaders.keys():
                if loaders[split] is None:
                    continue
                stacked_real_features = []
                self.print_fn(
                    f"Stacking Inception features for {split}:{self.n_samples} samples from the real dataset."
                )
                for _ in tqdm(range(num_batches)):
                    try:
                        real_samples = next(loaders[split])
                    except StopIteration:
                        break
                    if isinstance(real_samples, dict):
                        real_samples = real_samples['images']
                    real_samples = real_samples.to(self.device)
                    real_features = self.calculate_inception_features(real_samples)
                    stacked_real_features.append(real_features)
                features[split + "_real_features"] = torch.cat(stacked_real_features, dim=0).cpu()

            self.train_real_features = features["train_real_features"]
            self.test_real_features = features["test_real_features"]
            np.savez_compressed(path,
                                train_real_features=self.train_real_features.numpy(),
                                test_real_features=self.test_real_features.numpy())
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
        self.dataset_stats_loaded = True
        print("generated features for real dataset")

    def fid_score(self, grad=False):

        context = torch.no_grad() if not grad else NoContext()
        with context:
            if not self.dataset_stats_loaded:
                self.load_or_precalc_dataset_stats()
            self.sampler.eval()
            batches = num_to_groups(self.n_samples, self.batch_size)
            stacked_fake_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} generated samples."
            )
            for batch in tqdm(batches):
                fake_samples = self.sampler.sample(batch_size=batch)
                fake_features = self.calculate_inception_features(fake_samples)
                stacked_fake_features.append(fake_features)
            print("stacking features")
            stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu()
            print("features stacked")


        res = {}

        scores_fs = {
            'DIVERSITY': DIVERSITY(),
            'FID': FID(),
            'FLD': FLD(eval_feat="train"),
            # 'AuthPct': AuthPct(),
            # 'CTTest': CTTest(),
            'KID': KID(ref_feat='train'),
            # 'Precision': PrecisionRecall(mode='Precision'),
            # 'Recall': PrecisionRecall(mode='Recall'),
        }
        for s_name, fn in scores_fs.items():
            try:
                print(f"computing {s_name}...")
                res[s_name] = fn.compute_metric(stacked_fake_features, self.test_real_features, self.train_real_features)
                print(f"computed {s_name}!")
            except Exception as e:
                print(f"\nWARNING: score '{s_name}' could not be computed. \nException:\n{e.__class__.__name__}:{e}\n\n")
                # raise e

        return res


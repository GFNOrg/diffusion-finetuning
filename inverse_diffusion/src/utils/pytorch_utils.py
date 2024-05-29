from tqdm import tqdm

from models.classifiers import CNN, ResNet
from utils.data_loaders import get_dataset
from utils.diffusion import DiffTrainer, GaussianDiffusion
from utils.simple_io import *
from utils.visualization import plot_exp_logs

import torch as T
import numpy as np
import random

import torch.nn
import torch.nn.functional as F
import torch.nn as nn


def seed_experiment(seed):
    """Set the seed for reproducibility in PyTorch runs.

    Args:
        seed (int): The seed number.
    """
    # Set the seed for Python's 'random' module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    T.manual_seed(seed)

    # If using CUDA (PyTorch with GPU support)
    if T.cuda.is_available():
        T.cuda.manual_seed(seed)
        T.cuda.manual_seed_all(seed)  # for multi-GPU setups
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False


def maybe_detach(x, t, times_to_detach):
    return x.detach() if t in times_to_detach else x


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(T.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return T.cat([T.sin(x_proj), T.cos(x_proj)], dim=-1)


def train_classifier(model, dataset, data_path, patience=20, batch_size=64, image_size=32, channels=3, epochs=100,
                     multi_class_index=0, y_tensor=None, x_tensor=None, multinoise=False, scheduler=None, splits=['train', 'test'], workers=0, use_cuda=None):

    assert not multinoise or scheduler is not None, 'A scheduler is necessary to train a multinoise classifier, please provide one.'

    device = 'cuda' if use_cuda else 'cpu'

    net = nn.DataParallel(model).to(device)

    data = get_dataset(
        dataset=dataset,
        batch_size=batch_size,
        data_path=data_path,
        channels=channels,
        image_size=image_size,
        workers=workers,
        x_tensor=x_tensor,
        y_tensor=y_tensor,
        splits=splits,
        multi_class_index=multi_class_index
    )

    validate = True
    train_data, train_loader = data['train_data'], data['train_loader']
    if 'valid_data' in data.keys():
        valid_data, valid_loader = data['valid_data'], data['valid_loader']
    elif 'test_data' in data.keys():
        valid_data, valid_loader = data['test_data'], data['test_loader']
    else:
        validate = False

    patience_lv = 0
    best_params = None
    best_epoch = 0
    best_acc = None

    opt_classifier = T.optim.Adam(model.parameters())
    for epoch in range(epochs):

        train_loss = 0
        train_acc = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            if multinoise:
                t = torch.LongTensor(np.random.choice(scheduler.timesteps, x.shape[0]))
                x = scheduler.add_noise(x, torch.randn_like(x), t)  # noisify x

            logits = net(x)

            loss = F.cross_entropy(logits, y.flatten())
            acc = (logits.argmax(1) == y.flatten()).float().mean()
            train_loss += loss.item()
            train_acc += acc.item()

            opt_classifier.zero_grad()
            loss.backward()
            opt_classifier.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        valid_loss = 0.
        valid_acc = 0.
        if validate:
            valid_loss = 0
            valid_acc = 0
            for x, y in valid_loader:
                x = x.to(device) * 2 - 1
                y = y.to(device)

                if multinoise:
                    t = torch.LongTensor(np.random.choice(scheduler.timesteps, x.shape[0]))
                    x = scheduler.add_noise(x, torch.randn_like(x), t)  # noisify x

                logits = model(x)
                loss = F.cross_entropy(logits, y.flatten())
                acc = (logits.argmax(1) == y.flatten()).float().mean()
                valid_loss += loss.item()
                valid_acc += acc.item()

            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader)

            if best_acc is None or best_acc < valid_acc:
                best_params = model.state_dict()
                best_epoch = epoch
                best_acc = valid_acc
                patience_lv = 0
            else:
                patience_lv += 1

        print(f" epoch: {epoch}, patience: {patience_lv}/{patience},"
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:3f}, "
              f"valid_loss: {valid_loss * 100:.4f}, valid_acc: {valid_acc * 100:3f}")

        if patience_lv >= patience:
            break

    return best_params, best_epoch, model


# def zero_burn_in(policy, train_dataset, args, transforms=None, y_class='labels', burn_in_iterations=5000):
#
#     # biw = BurnInWrapper(
#     #     policy,
#     #     image_size=args.image_size,
#     #     timesteps=args.traj_length,  # number of steps
#     #     sampling_timesteps=args.sampling_length
#     # )
#     biw = GaussianDiffusion(
#         policy,
#         image_size=args.image_size,
#         timesteps=args.traj_length,  # number of steps
#         sampling_timesteps=args.sampling_length
#     )
#
#     trainer = DiffTrainer(
#         biw,
#         train_dataset,
#         None,
#         logger,
#         results_folder=args.save_folder,
#         train_batch_size=args.batch_size,
#         train_lr=1e-4,
#         amp=args.mixed_precision,
#         train_num_steps=burn_in_iterations,
#         workers=args.workers,
#         dataset_name=args.dataset,
#         data_path=args.data_path,
#         num_samples=args.plot_batch_size,
#         device=args.device,
#         show_figures=args.show_figures,
#         save_figures=args.save_figures
#     )
#
#     trained_policy = trainer.train()
#     return trained_policy
#
#
# def diffusion_burn_in(policy, train_dataset, args, transforms=None, y_class='labels', burn_in_iterations=1000):
#
#     gd = GaussianDiffusion(
#         policy,
#         image_size=args.image_size,
#         timesteps=args.traj_length,  # number of steps
#         sampling_timesteps=args.sampling_length
#     )
#
#     trainer = DiffTrainer(
#         gd,
#         train_dataset,
#         None,
#         logger=logger,
#         results_folder=args.save_folder,
#         train_batch_size=args.batch_size,
#         train_lr=1e-4,
#         amp=args.mixed_precision,
#         train_num_steps=burn_in_iterations,
#         workers=args.workers,
#         num_samples=args.plot_batch_size,
#         device=args.device,
#         dataset_name=args.dataset,
#         data_path=args.data_path,
#         show_figures=args.show_figures,
#         save_figures=args.save_figures
#     )
#
#     trained_policy = trainer.train()
#     return trained_policy


class Logger:

    def __init__(self, args=None):
        self.args = args
        self.logs = {}

    def log(self, results):
        for key, item in results.items():
            if key not in self.logs.keys():
                self.logs[key] = []
            else:
                if isinstance(results[key], torch.Tensor):
                    self.logs[key].append(results[key].mean().item())
                else:
                    self.logs[key].append(results[key])

    def show(self, args, save=False):
        plot_exp_logs(self.logs, self.args, show=True, path=self.args.save_folder, save=save)

    def print(self, it=None):
        it = it if it is not None else ''
        print(f"it {it}: " + ", ".join([f"{key}: {np.mean(value[-10:]):.4f}" for key, value in self.logs.items()]))

    def save(self):
        save_dict_to_file(data=self.logs, path=self.args.save_folder, filename='run_logs', format='json', replace=True)

    def save_args(self):
        save_dict_to_file(data=self.args.__dict__, path=self.args.save_folder, filename='run_args', format='json', replace=True)

    def load(self, path):
        self.args = load_dict_from_file(f"{path}/run_args.json")
        self.logs = load_dict_from_file(f"{path}/run_logs.json")


def print_gpu_memory(total_memory=None, memory_allocated=None, memory_free=None):
    current_memory_free, current_total_memory = torch.cuda.mem_get_info()
    current_memory_allocated = current_total_memory - current_memory_free

    if total_memory is None or memory_allocated is None or memory_free is None:
        print()
        print(f"Total memory: {current_total_memory / (1024 ** 3):.4} GB")
        print(f"Used memory: {current_memory_allocated / (1024 ** 3):.4f} GB")
        print(f"Free memory: {current_memory_free / (1024 ** 3):.4f} GB")

    else:
        print(f"Total memory change: {(current_total_memory - total_memory) / (1024 ** 3):.4f} GB")
        print(f"Used memory change: {(current_memory_allocated - memory_allocated) / (1024 ** 3):.4f} GB")
        print(f"Free memory change: {(current_memory_free - memory_free) / (1024 ** 3):.4f} GB")
    return current_total_memory, current_memory_allocated, current_memory_free


def get_gpu_memory():
    current_total_memory = T.cuda.get_device_properties(0).total_memory
    current_memory_allocated = T.cuda.memory_allocated()
    current_memory_free = T.cuda.memory_reserved() - T.cuda.memory_allocated()
    return current_total_memory, current_memory_allocated, current_memory_free


class NoContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_train_classifier(args, scheduler=None):
    """retrieve a classifier if there is one, else train one"""

    if 'cnn' in args.classifier_model.lower():
        classifier = CNN(
            input_size=args.image_size,
            channels=args.channels,
            num_classes=args.num_classes
        )
    else:
        classifier = ResNet(
            input_size=args.image_size,
            channels=args.channels,
            num_classes=args.num_classes,
            depth=args.classifier_depth,
            finetune=args.classifier_pretrained
        )

    classifier_name_requirements = [args.dataset, "classifier", f"_{args.multi_class_index}_"]
    # if args.method in ['dp', 'lgd_mc']:
    #     classifier_name_requirements.append('multinoise')

    pretrained_classifer = get_filenames(
        args.load_path,
        contains=classifier_name_requirements,
        ends_with='.pth'
    )

    if len(pretrained_classifer) == 0:
        print(f"No classifier '{args.model}' found for '{args.dataset}' data, \nTraining classifier...")
        trained_params, best_epoch, classifier = train_classifier(
            model=classifier,
            batch_size=128,
            data_path=args.data_path,
            dataset=args.dataset,
            channels=args.channels,
            image_size=args.image_size,
            x_tensor=args.x_tensor,
            y_tensor=args.y_tensor,
            multinoise=args.method in ['dp', 'lgd_mc'],
            scheduler=scheduler,
            multi_class_index=args.multi_class_index,
            workers=0 if args is None or 'linux' not in args.system else 4,
            use_cuda=args.use_cuda
        )
        name_base = '_'.join(classifier_name_requirements)
        T.save(trained_params, f'{args.load_path}/{name_base}_{best_epoch}.pth')
    else:
        classifier.load_state_dict(
            T.load(f'{args.load_path}/{pretrained_classifer[-1]}', map_location=torch.device('cpu')))
        print("Loaded existing classifier!")

    print(f'Total params: \nclassifier: {(sum(p.numel() for p in classifier.parameters()) / 1e6):.2f}M ')

    classifier = nn.DataParallel(classifier).to(args.device)
    classifier.eval()

    for p in classifier.parameters():
        p.requires_grad = False  # freeze classifier

    return classifier

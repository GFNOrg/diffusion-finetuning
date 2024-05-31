import time

import deeplake
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from utils.simple_io import *
from torchvision import transforms
from PIL import Image
import torch
from pathlib import Path

import re

DATASETS = {
    'mnist': {
        'splits': ['train', 'test'],
        'image_size': 32,
        'noise_size': 32,
        'channels': 1,
        'num_classes': 10,
        'x_tensor': 'images',
        'y_tensor': 'labels',
        'multi_class_index': 0,
        'finetune_class': 0,
        'block_out_channels': (32, 64, 128, 256),
        'classifier_model': 'cnn',
        'classifier_depth': 10,
        'fid_inception_block': 768,  # 64, 192, 768, 2048
        'train_path': "",
        'test_path': "",
    },
    'cifar10': {
        'splits': ['train', 'test'],
        'image_size': 32,
        'noise_size': 32,
        'channels': 3,
        'num_classes': 10,
        'x_tensor': 'images',
        'y_tensor': 'labels',
        'multi_class_index': 0,
        'finetune_class': 0,
        'block_out_channels': (32, 64, 128, 256),
        'classifier_model': 'resnet',
        'classifier_depth': 18,
        'fid_inception_block': 768,  # 64, 192, 768, 2048
        'train_path': "",
        'test_path': ""
    },
    'CelebA': {
        'splits': ['train', 'test'],
        'image_size': 128,
        'noise_size': 128,
        'num_classes': 2,
        'channels': 3,
        'x_tensor': 'images',
        'y_tensor': 'mustache',
        'multi_class_index': 0,
        'finetune_class': 1,
        'block_out_channels': (32, 64, 128, 256),
        'classifier_model': 'resnet',
        'classifier_depth': 18,
        'fid_inception_block': 768, # 64, 192, 768, 2048
        'train_path': "",
        'test_path': ""
    },
    'dsprites': {
        'splits': None,
        'channels': 1,
        'image_size': 32,
        'noise_size': 32,
        'num_classes': 4,
        'x_tensor': 'images',
        'y_tensor': 'latents_values',
        'multi_class_index': 0,
        'finetune_class': 0,
        'block_out_channels': (32, 64, 128, 256),
        'classifier_model': 'resnet',
        'classifier_depth': 18,
        'fid_inception_block': 768,  # 64, 192, 768, 2048
        'train_path': "",
        'test_path': ""
    },
    'UTKFace': {
        'splits': ['train', 'test'],
        'image_size': 64,
        'noise_size': 64,
        'channels': 3,
        'num_classes': 3,
        'x_tensor': 'images',
        'y_tensor': 'n/a',
        'multi_class_index': 0,
        'finetune_class': 0,
        'block_out_channels': (32, 64, 128, 256),
        'classifier_model': 'resnet',
        'classifier_depth': 18,
        'fid_inception_block': 768,  # 64, 192, 768, 2048
        'train_path': "UTKFace",  # local
        'test_path': "crop_part1"  # local
    },
    'domainnet-quickdraw': {
        'splits': ['train', 'test'],
        'image_size': 64,
        'noise_size': 64,
        'channels': 3,
        'num_classes': 2,
        'x_tensor': 'image',
        'y_tensor': 'object',
        'multi_class_index': 0,
        'finetune_class': 0,
        'block_out_channels': (32, 64, 128, 256),
        'classifier_model': 'resnet',
        'classifier_depth': 18,
        'fid_inception_block': 768,  # 64, 192, 768, 2048
        'train_path': "",  # local
        'test_path': ""  # local
    },
}
CELEB_CLASSES = [
    'clock_shadow',
    'arched_eyebrows',
    'attractive',
    'bags_under_eyes',
    'bald',
    'bangs',
    'big_lips',
    'big_nose',
    'black_hair',
    'blond_hair',
    'blurry',
    'brown_hair',
    'bushy_eyebrows',
    'chubby',
    'double_chin',
    'eyeglasses',
    'goatee',
    'gray_hair',
    'heavy_makeup',
    'high_cheekbones',
    'male',
    'mouth_slightly_open',
    'mustache',
    'narrow_eyes',
    'no_beard',
    'oval_face',
    'pale_skin',
    'pointy_nose',
    'receding_hairline',
    'rosy_cheeks',
    'sideburns',
    'smiling',
    'straight_hair',
    'wavy_hair',
    'wearing_earrings',
    'wearing_hat',
    'wearing_lipstick',
    'wearing_necklace',
    'wearing_necktie',
    'young',
]


class ScaleTransform:
    """Scale the tensor values."""
    def __call__(self, tensor):
        return (tensor * 2) - 1


class CustomDataset(Dataset):
    def __init__(self, folder, image_size=None, exts=['jpg', 'jpeg', 'png'], transforms=None, dataset='mnist',
                 channels=3, cls_idx=0):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.channels = channels
        self.dataset_name = dataset
        self.cls_idx = cls_idx
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def _parse_y(self, path):
        if 'face' in self.dataset_name.lower():
            # extracts class
            try:
                split = re.split('\\\\|/', str(path))[-1].split('_')
                if '20170116174525125' in split[-1]:
                    split = ['39', '0', '1', split[-1]]
                elif '20170109150557335' in split[-1]:
                    split = ['61', '1', '3', split[-1]]
                elif '20170109142408075' in split[-1]:
                    split = ['61', '1', '1', split[-1]]
                return torch.LongTensor([int(split[self.cls_idx])])  # try to extract class
            except:
                return torch.LongTensor([-1])  # others
        else:
            return torch.LongTensor([int(re.split('\\\\|/', str(path))[-2])])

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        if self.channels == 3:
            img = img.convert('RGB')
        y = self._parse_y(path)
        # todo: add special case for UTKFace

class CustomDatasetDL(Dataset):
    def __init__(self, dt, image_size=None, perturb=True, flip=True, dataset='mnist', channels=3, y_tensor=None):
        super().__init__()
        self.ds = dt
        self.image_size = image_size
        self.channels = channels
        self.dataset_name = dataset
        self.y_tensor = 'labels' if y_tensor is None else y_tensor

        tfs = []
        if image_size is not None:
            tfs += [transforms.ToPILImage()]
            if perturb:
                tfs += [transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
                        transforms.RandomCrop(image_size)]
            if flip:
                tfs += [transforms.RandomHorizontalFlip()]
        tfs += [transforms.ToTensor(),
                ScaleTransform()]

        self.transform = transforms.Compose(tfs)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):

        img = self.ds.images[index].numpy(fetch_chunks=False)

        if len(img.shape) == 2:
            img = img.reshape(img.shape + (1,))
            if self.channels == 3:
                img = np.concatenate([img, img, img], axis=-1)

        y = torch.LongTensor(getattr(self.ds, self.y_tensor)[index].numpy(fetch_chunks=False).astype(np.int32))

        return self.transform(img), y


def get_dataset(
    dataset,
    batch_size,
    data_path,
    channels=3,
    image_size=32,
    workers=0,
    x_tensor='images',
    y_tensor='labels',
    splits=['train', 'test'],
    multi_class_index=0,
    filter_data=False,
    filter_labels=None
    ):
    data = {}

    if 'celeb' in dataset.lower():
        dt_id = 'celeb-a'
    else:
        dt_id = dataset.lower()

    tfs = []
    if image_size is not None:
        tfs += [transforms.ToPILImage(),
                transforms.Resize((image_size, image_size))]

        if 'mnist' not in dataset.lower() and 'dsprites' not in dataset.lower():
            tfs += [transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip()]
    tfs += [transforms.ToTensor(),
            ScaleTransform()]

    if 'utkface' in dataset.lower():
        for split in splits:
            data[f'{split}_data'] = CustomDataset(f"{data_path}/{DATASETS[dataset][f'{split}_path']}",
                                                  image_size=image_size,
                                                  transforms=transforms.Compose(tfs[1:]),
                                                  dataset=dataset,
                                                  channels=channels,
                                                  cls_idx=multi_class_index)

            data[f'{split}_loader'] = DataLoader(data[f'{split}_data'],
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=workers)
    else:

        if splits is None:
            splits = ['train']
            dataset_id = dt_id
        else:
            dataset_id = None

        for split in splits:
            if dataset_id is None:
                dataset_id = f"{dt_id}-{split}"

            # use deeplake
            data[f'{split}_data'] = deeplake.dataset(f"hub://activeloop/{dataset_id}",
                access_method='local',
                reset=True
            )
            if data[f'{split}_data'].has_head_changes:
                data[f'{split}_data'].commit()

            if filter_data and filter_labels is not None:
                view_id = f'{dataset.lower()}_{y_tensor}_{split}_{tuple(filter_labels)}'
                if view_id in [view.id for view in data[f'{split}_data'].get_views()]:
                    print(f"Found view for classes {y_tensor}: {filter_labels} ({split})")
                else:
                    print(f"No local view was found for classes {y_tensor}:{filter_labels} ({split}), creating one...")
                    data[f'{split}_data'].filter(
                        filter_fun(y_tensor, filter_labels),
                        scheduler='threaded',
                        num_workers=workers,
                        save_result=True,
                        result_ds_args={
                            'id': view_id,
                            'message': f"Samples of {split} with {y_tensor} in '{filter_labels}'",
                        }
                    )

                class_view = data[f'{split}_data'].load_view(view_id)
                print(f"Number of samples for {y_tensor}:{filter_labels} in {split} is {len(class_view)}")

                data[f'{split}_class_view'] = class_view.pytorch(
                    batch_size=batch_size,
                    num_workers=workers,
                    transform={x_tensor: transforms.Compose(tfs),
                               y_tensor: None},
                    shuffle=True
                )

            data[f'{split}_loader'] = data[f'{split}_data'].pytorch(
                batch_size=batch_size,
                num_workers=workers,
                transform={x_tensor: transforms.Compose(tfs),
                           y_tensor: None},
                shuffle=True
            )
            dataset_id = None

    return data

@deeplake.compute
def filter_fun(sample_in, y_tensor, labels_list):
    return sample_in[y_tensor].data()['value'][0] in labels_list


def cycle(dl):
    while True:
        for data in dl:
            if isinstance(data, dict):
                yield data['images']
            else:
                yield data

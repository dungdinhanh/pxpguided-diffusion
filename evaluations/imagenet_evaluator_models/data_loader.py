import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import hfai
import hfai.datasets
from torch.utils.data.distributed import DistributedSampler
import  numpy as np
from ffrecord.torch import Dataset, DataLoader
from hfai.datasets.base import (
    BaseDataset,
    get_data_dir,
)
from typing import Callable, Optional
from PIL import Image

import pickle
import math



def data_loader(batch_size=256, workers=4, pin_memory=True, mini=False, img_size=64):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = hfai.datasets.ImageNet(split='train', transform=train_transform, miniset=mini)
    train_datasampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = train_dataset.loader(batch_size, sampler=train_datasampler,
                                            num_workers=workers, pin_memory=pin_memory)

    val_dataset = hfai.datasets.ImageNet(split='val', transform=val_transform, miniset=mini)
    val_datasampler = DistributedSampler(val_dataset)
    val_loader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=workers,
                                        pin_memory=pin_memory)
    return train_loader, val_loader


def data_loader_gd(batch_size=256, workers=4, pin_memory=True, mini=False, img_size=64):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = ImageNetHF2(img_size, random_crop=True, random_flip=True, split='train', classes=True,
                             miniset=mini, transform=train_transform)
    train_datasampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = train_dataset.loader(batch_size, sampler=train_datasampler,
                                            num_workers=workers, pin_memory=pin_memory)

    val_dataset = ImageNetHF2(img_size, random_crop=False, random_flip=False, split='val', classes=True,
                             miniset=mini, transform=val_transform)
    val_datasampler = DistributedSampler(val_dataset)
    val_loader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=workers,
                                        pin_memory=pin_memory)
    return train_loader, val_loader


class ImageNetHF2(hfai.datasets.ImageNet):
    def __init__(self, resolution, random_crop=False, random_flip=True, split='train', classes=True, miniset=False, transform=None):
        # to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        super(ImageNetHF2, self).__init__(split=split, transform=None, check_data=True, miniset=miniset)
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_classes = classes
        self.transform = transform

    def __getitem__(self, indices):
        imgs_bytes = self.reader.read(indices)
        samples = []
        for i, bytes_ in enumerate(imgs_bytes):
            img = pickle.loads(bytes_).convert("RGB")
            label = self.meta["targets"][indices[i]]
            samples.append((img, int(label)))

        transformed_samples = []
        for img, label in samples:
            if self.random_crop:
                arr = random_crop_arr(img, self.resolution)
            else:
                arr = center_crop_arr(img, self.resolution)

            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]

            # img = arr.astype(np.float32) / 127.5 - 1
            # img = np.transpose(img, [2, 0, 1]) # might not need to transpose
            if self.transform is not None:
                img = self.transform(arr.copy())
            out_dict = None
            if self.local_classes:
                out_dict = label

            transformed_samples.append((img, out_dict))
        return transformed_samples

def val_imagenet_loader(batch_size=256, workers=4, pin_memory=True, mini=False, diff_transform=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if not diff_transform:
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print("diffusion transform")
        val_transform = diffuse_transform


    val_dataset = hfai.datasets.ImageNet(split='val', transform=val_transform, miniset=mini)
    val_datasampler = DistributedSampler(val_dataset)
    val_loader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=workers,
                                    pin_memory=pin_memory)
    return val_loader

def val_imagenet_loader_lowres(batch_size=256, workers=4, pin_memory=True, mini=False, diff_transform=False, res=256):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if res== 256:
        cres = 224
    else:
        cres = res
    if not diff_transform:
        val_transform = transforms.Compose([
            transforms.Resize(res),
            transforms.CenterCrop(cres),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print("diffusion transform")
        val_transform = diffuse_transform


    val_dataset = hfai.datasets.ImageNet(split='val', transform=val_transform, miniset=mini)
    val_datasampler = DistributedSampler(val_dataset)
    val_loader = val_dataset.loader(batch_size, sampler=val_datasampler, num_workers=workers,
                                    pin_memory=pin_memory)
    return val_loader

class SampleDataset(BaseDataset):
    def __init__(self, path, transform: Optional[Callable] = None):
        super(SampleDataset, self).__init__()
        data_npz = np.load(path)
        # if test:
        #     self.samples = data_npz['arr_0'][:10]
        #     self.labels = data_npz['arr_1'][:10]
        # else:
        self.samples = data_npz['arr_0']
        self.labels = data_npz['arr_1']
        self.transform = transform

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, indices):
        samples = self.samples[indices]
        labels = self.labels[indices]
        samples_labels = []

        for i, sample in enumerate(samples):
            img = Image.fromarray(sample, "RGB")
            cls = labels[i]
            samples_labels.append((img, cls))
        # samples_labels = zip(samples, labels)

        transformed_samples = []

        for sample, label in samples_labels:
            if self.transform:
                sample = self.transform(sample)
            transformed_samples.append((sample, label))
        return transformed_samples
        pass


def eval_loader(batch_size=256, pin_memory=True, path: str="runs/test.npz",  diff_transform=False):
    assert os.path.isfile(path), f"no checkpoints found at {path}"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if not diff_transform:
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print("diffusion transform")
        val_transform = diffuse_transform

    samples_dataset = SampleDataset(path=path, transform=val_transform)
    samples_datasampler = DistributedSampler(samples_dataset, shuffle=False)
    sample_loader = samples_dataset.loader(batch_size, sampler=samples_datasampler, num_workers=1,
                                    pin_memory=pin_memory)
    return sample_loader

import random
def diffuse_transform(img):
    arr = center_crop_arr(img, 256)

    if  random.random() < 0.5:
        arr = arr[:, ::-1]

    img = arr.astype(np.float32) / 127.5 - 1
    img = np.transpose(img, [2, 0, 1])
    return img



def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

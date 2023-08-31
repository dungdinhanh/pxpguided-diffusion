import copy
import math
import os
import random
from PIL import Image
import io
import blobfile as bf
from hfai.datasets.base import (
    BaseDataset,
    get_data_dir,
    register_dataset)
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
import numpy as np
from torch.utils.data import  Dataset
from hfai.datasets.imagenet import ImageNet
from torchvision.datasets.mnist import read_image_file, read_label_file
import warnings
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
from torchvision.datasets.celeba import CSV
from torchvision.transforms import transforms
from collections import namedtuple
import csv
from functools import partial
import torch
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
import torchvision



class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class ImageNetHF(ImageNet):
    def __init__(self, resolution, random_crop=False, random_flip=True, split='train', classes=True, miniset=False):
        # to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        super(ImageNetHF, self).__init__(split=split, transform=None, check_data=True, miniset=miniset)
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_classes = classes

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

            img = arr.astype(np.float32) / 127.5 - 1
            img = np.transpose(img, [2, 0, 1]) # might not need to transpose
            # if self.transform:
            #     img = self.transform(img)
            out_dict = {}
            if self.local_classes:
                out_dict["y"] = label

            transformed_samples.append((img, out_dict))
        return transformed_samples

class ImageNetHFLocal(ImageNet):
    def __init__(self, resolution, random_crop=False, random_flip=True, split='train', classes=True):
        # to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        # super(ImageNetHFLocal, self).__init__(split=split, transform=None, check_data=True, miniset=True)
        super(ImageNet, self).__init__()
        miniset =  True
        check_data = True
        assert split in ["train", "val"]
        self.split = split
        self.transform = None
        data_dir = Path("data/imagenet")
        if miniset:
            data_dir = data_dir / "mini"
        self.data_dir = data_dir / "ImageNet"
        self.fname = self.data_dir / f"{split}.ffr"
        self.reader = FileReader(self.fname, check_data)

        with open(self.data_dir / f"{split}.ffr" / "meta.pkl", "rb") as fp:
            self.meta = pickle.load(fp)
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_classes = classes

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

            img = arr.astype(np.float32) / 127.5 - 1
            img = np.transpose(img, [2, 0, 1]) # might not need to transpose
            # if self.transform:
            #     img = self.transform(img)
            out_dict = {}
            if self.local_classes:
                out_dict["y"] = label

            transformed_samples.append((img, out_dict))
        return transformed_samples




class MNISTHF(BaseDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

        Args:
            root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
                and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
            train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
                otherwise from ``t10k-images-idx3-ubyte``.
            download (bool, optional): If True, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """

    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            classes=False, split='train'
    ) -> None:
        super().__init__()
        train = (split=='train')
        self.local_classes = classes
        root = "./data"
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        download = True
        self.transform = None
        # has_transforms = False
        # has_separate_transform = transform is not None or target_transform is not None
        # if has_transforms and has_separate_transform:
        #     raise ValueError("Only transforms or transform/target_transform can be passed as argument")
        #
        # # for backwards-compatibility
        # self.transform = transform
        # self.target_transform = target_transform

        # if has_separate_transform:
        # transforms = StandardTransform(transform, target_transform)
        # self.transforms = transforms
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def __getitem__(self, indices):
        imgs, targets = self.data[indices].numpy(), self.targets[indices].numpy()
        no_images = len(targets)
        transformed_samples = []
        for i in range(no_images):
            img = imgs[i]
            target = int(targets[i])
            # img = Image.fromarray(img.numpy(), mode="L")
            if self.transform is not None:
                img = self.transform(img)
            img = img/127.5 - 1
            img = np.expand_dims(img, axis=0)
            img = torch.Tensor(img).float()
            out_dict = {}
            if self.local_classes:
                out_dict["y"] = target
            transformed_samples.append((img, out_dict))
        return transformed_samples


from datasets.ffrecord_support import *


class CelebAHF(BaseDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        # ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            resolution,
            random_crop=False,
            random_flip=True,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            classes=False
    ) -> None:
        super(CelebAHF, self).__init__()
        image_file = os.path.join(get_data_dir(), "/private_dataset/CelebA/celeba.ffr")
        self.root = root
        # read from celeba.ffr
        self.reader_celeb = PackedFolder(image_file)
        self.images_folder = "img_align_celeba"
        #
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])
        self.split = split
        self.classes = classes
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        self.filename = splits.index
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                print(fpath)
                return False
        return True
        # Should check a hash of the images
        # return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def _get_paths_ff(self, indices):
        list_paths = []
        for index in indices:
            list_paths.append(os.path.join("img_align_celeba", self.filename[index]))
        return list_paths
    def __getitem__(self, indices):
        list_bytes = self.reader_celeb.read(self._get_paths_ff(indices))
        transformed_samples = []
        for i in range(len(list_bytes)):
            img = PIL.Image.open(io.BytesIO(list_bytes[i]))
            # if self.random_crop:
            #     arr = random_crop_arr(img, self.resolution)
            # else:
            #     arr = center_crop_arr(img, self.resolution)
            #
            # if self.random_flip and random.random() < 0.5:
            #     arr = arr[:, ::-1]
            #
            # img = arr.astype(np.float32) / 127.5 - 1
            # img = np.transpose(img, [2, 0, 1])  # might not need to transpose
            if self.transform:
                img = self.transform(img)
            out_dict = {}
            if self.classes:
                target: Any = []
                for t in self.target_type:
                    if t == "attr":
                        target.append(self.attr[indices[i], :])
                    elif t == "identity":
                        target.append(self.identity[indices[i], 0])
                    elif t == "bbox":
                        target.append(self.bbox[indices[i], :])
                    elif t == "landmarks":
                        target.append(self.landmarks_align[indices[i], :])
                    else:
                        # TODO: refactor with utils.verify_str_arg
                        raise ValueError("Target type \"{}\" is not recognized.".format(t))

                if target:
                    target = tuple(target) if len(target) > 1 else target[0]

                    # if self.target_transform is not None:
                    #     target = self.target_transform(target)
                    out_dict['y'] = target
                else:
                    target = None
            transformed_samples.append((img, out_dict))
        return transformed_samples

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class CelebA64HF(BaseDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        # ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            resolution,
            random_crop=False,
            random_flip=True,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            classes=False
    ) -> None:
        super(CelebA64HF, self).__init__()
        image_file = os.path.join(get_data_dir(), "/private_dataset/CelebA/celeba.ffr")
        self.root = root
        # read from celeba.ffr
        self.reader_celeb = PackedFolder(image_file)
        self.images_folder = "img_align_celeba"
        #
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.transform=None
        self.split = split
        self.classes = classes
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        self.filename = splits.index
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                print(fpath)
                return False
        return True
        # Should check a hash of the images
        # return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def _get_paths_ff(self, indices):
        list_paths = []
        for index in indices:
            list_paths.append(os.path.join("img_align_celeba", self.filename[index]))
        return list_paths
    def __getitem__(self, indices):
        list_bytes = self.reader_celeb.read(self._get_paths_ff(indices))
        transformed_samples = []
        for i in range(len(list_bytes)):
            img = PIL.Image.open(io.BytesIO(list_bytes[i]))
            if self.random_crop:
                arr = random_crop_arr(img, self.resolution)
            else:
                arr = center_crop_arr(img, self.resolution)

            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]

            img = arr.astype(np.float32) / 127.5 - 1
            img = np.transpose(img, [2, 0, 1])  # might not need to transpose
            # if self.transform:
            #     img = self.transform(img)
            out_dict = {}
            if self.classes:
                target: Any = []
                for t in self.target_type:
                    if t == "attr":
                        target.append(self.attr[indices[i], :])
                    elif t == "identity":
                        target.append(self.identity[indices[i], 0])
                    elif t == "bbox":
                        target.append(self.bbox[indices[i], :])
                    elif t == "landmarks":
                        target.append(self.landmarks_align[indices[i], :])
                    else:
                        # TODO: refactor with utils.verify_str_arg
                        raise ValueError("Target type \"{}\" is not recognized.".format(t))

                if target:
                    target = tuple(target) if len(target) > 1 else target[0]

                    # if self.target_transform is not None:
                    #     target = self.target_transform(target)
                    out_dict['y'] = target
                else:
                    target = None
            transformed_samples.append((img, out_dict))
        return transformed_samples

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class CelebA32UPHF(BaseDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        # ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            resolution,
            random_crop=False,
            random_flip=True,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            classes=False
    ) -> None:
        super(CelebA32UPHF, self).__init__()
        image_file = os.path.join(get_data_dir(), "/private_dataset/CelebA/celeba.ffr")
        self.root = root
        # read from celeba.ffr
        self.reader_celeb = PackedFolder(image_file)
        self.images_folder = "img_align_celeba"
        #
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.transform=None
        self.split = split
        self.classes = classes
        self.transform32 = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        self.filename = splits.index
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                print(fpath)
                return False
        return True
        # Should check a hash of the images
        # return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def _get_paths_ff(self, indices):
        list_paths = []
        for index in indices:
            list_paths.append(os.path.join("img_align_celeba", self.filename[index]))
        return list_paths
    def __getitem__(self, indices):
        list_bytes = self.reader_celeb.read(self._get_paths_ff(indices))
        transformed_samples = []
        for i in range(len(list_bytes)):
            img = PIL.Image.open(io.BytesIO(list_bytes[i]))

            img32 = self.transform32(img)

            if self.random_crop:
                arr = random_crop_arr(img, self.resolution)
            else:
                arr = center_crop_arr(img, self.resolution)

            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]

            img = arr.astype(np.float32) / 127.5 - 1
            img = np.transpose(img, [2, 0, 1])  # might not need to transpose
            # if self.transform:
            #     img = self.transform(img)
            out_dict = {}
            if self.classes:
                target: Any = []
                for t in self.target_type:
                    if t == "attr":
                        target.append(self.attr[indices[i], :])
                    elif t == "identity":
                        target.append(self.identity[indices[i], 0])
                    elif t == "bbox":
                        target.append(self.bbox[indices[i], :])
                    elif t == "landmarks":
                        target.append(self.landmarks_align[indices[i], :])
                    else:
                        # TODO: refactor with utils.verify_str_arg
                        raise ValueError("Target type \"{}\" is not recognized.".format(t))

                if target:
                    target = tuple(target) if len(target) > 1 else target[0]

                    # if self.target_transform is not None:
                    #     target = self.target_transform(target)
                    out_dict['y'] = target
                else:
                    target = None
            transformed_samples.append((img32, img, out_dict))
        return transformed_samples

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

class CelebALocal(BaseDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            resolution,
            random_crop=False,
            random_flip=True,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            classes=False
    ) -> None:
        super(CelebALocal, self).__init__()
        download = True
        self.root = root
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])
        self.split = split
        self.classes = classes
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        self.filename = splits.index
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, indices):
        transformed_samples = []
        for index in indices:
            img = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
            # if self.random_crop:
            #     arr = random_crop_arr(img, self.resolution)
            # else:
            #     arr = center_crop_arr(img, self.resolution)
            #
            # if self.random_flip and random.random() < 0.5:
            #     arr = arr[:, ::-1]
            #
            # img = arr.astype(np.float32) / 127.5 - 1
            # img = np.transpose(img, [2, 0, 1])  # might not need to transpose
            if self.transform:
                img = self.transform(img)
            out_dict = {}
            if self.classes:
                target: Any = []
                for t in self.target_type:
                    if t == "attr":
                        target.append(self.attr[index, :])
                    elif t == "identity":
                        target.append(self.identity[index, 0])
                    elif t == "bbox":
                        target.append(self.bbox[index, :])
                    elif t == "landmarks":
                        target.append(self.landmarks_align[index, :])
                    else:
                        # TODO: refactor with utils.verify_str_arg
                        raise ValueError("Target type \"{}\" is not recognized.".format(t))

                if target:
                    target = tuple(target) if len(target) > 1 else target[0]

                    # if self.target_transform is not None:
                    #     target = self.target_transform(target)
                    out_dict['y'] = target
                else:
                    target = None
            transformed_samples.append((img, out_dict))
        return transformed_samples

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class CIFAR10HF(torchvision.datasets.CIFAR10, BaseDataset):
    """
    这是一个用于识别普适物体的小型数据集

    该数据集一共包含 10 个类别的 RGB 彩色图片，每个图片的尺寸为 32 × 32 ，每个类别有 600 个图像，数据集中一共有 500 张训练图片和 100 张测试图片。更多信息参考官网：https://www.cs.toronto.edu/~kriz/cifar.html

    Args:
        split (str): 数据集划分形式，包括：训练集（``train``）或者验证集（``val``）
        transform (Callable): transform 函数，对图片进行 transfrom，接受一张图片作为输入，输出 transform 之后的图片
        target_transform (Callable): 对 target 进行 transfrom，接受一个 target 作为输入，输出 transform 之后的 target

    Returns:
        image, target (PIL.Image.Image, int): 返回的每条样本是一个元组，包含一个RGB格式的图片，及其对应的目标标签

    Examples:

    .. code-block:: python

        from hfai.datasets import CIFAR10
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        dataset = CIFAR10('train', transform)
        loader = dataset.loader(batch_size=64, num_workers=4)

        for image, target in loader:
            # training model

    NOTE:
        使用的时候所有数据会直接加载进内存，大小大约为 178 MiB。``CIFAR10`` 和 ``CIFAR100`` 的 ``loader()`` 方法返回的是一个 ``torch.utils.data.DataLoader`` ，而不是 ``ffrecord.torch.DataLoader`` 。

    """

    def __init__(
        self, split: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, data_folder=None,
            local_classes=True, random_flip=True
    ) -> None:
        assert split in ["train", "test"]
        if data_folder is None:
            data_folder = str(get_data_dir())
        super().__init__(os.path.join(data_folder, "CIFAR"), split == "train", transform, target_transform, download=True)
        self.resolution=32
        self.local_classes = local_classes
        self.random_flip = random_flip

    def loader(self, *args, **kwargs) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, *args, **kwargs)


    def __getitem__(self, indices):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        transformed_samples = []
        for index in indices:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)
            arr = center_crop_arr(img, self.resolution)
            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]
            img = arr.astype(np.float32) / 127.5 - 1
            img = np.transpose(img, [2, 0, 1])
            out_dict = {}
            if self.local_classes:
                out_dict["y"] = target

            transformed_samples.append((img, out_dict))
        return transformed_samples

    def __len__(self) -> int:
        return len(self.data)


class ImageNetHFtest(ImageNet):
    def __init__(self, resolution, random_crop=False, random_flip=True, split='train', classes=True):
        # to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        super(ImageNetHFtest, self).__init__(split=split, transform=None, check_data=True, miniset=False)
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.local_classes = classes

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

            img = arr.astype(np.float32) / 127.5 - 1
            img = np.transpose(img, [2, 0, 1]) # might not need to transpose
            # if self.transform:
            #     img = self.transform(img)
            out_dict = {}
            if self.local_classes:
                out_dict["y"] = label

            transformed_samples.append((img, out_dict))
        return transformed_samples

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
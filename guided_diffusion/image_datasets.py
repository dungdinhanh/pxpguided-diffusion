from torch.utils.data.distributed import DistributedSampler
from ffrecord.torch import DataLoader
from guided_diffusion.vision_images import *


def load_data_imagenet_hfai(
    *,
    train=True,
    batch_size,
    image_size,
    random_crop=False,
    random_flip=True,
    class_cond=True,
        miniset=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    # classes = None
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    #     random_crop=random_crop,
    #     random_flip=random_flip,
    # )
    if train:
        dataset = ImageNetHF(image_size, random_crop=random_crop, random_flip=random_flip, split='train', classes=class_cond,
                             miniset=miniset)
    else:
        dataset = ImageNetHF(image_size, random_crop=random_crop, random_flip=random_flip, split='val', classes=class_cond,
                             miniset=miniset)
    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader # put all items of loader into list and concat all list infinitely
    # return loader

def load_data_imagenet_hfai_localmini(*,
    train=True,
    batch_size,
    image_size,
    random_crop=False,
    random_flip=True,
    class_cond=True):
    if train:
        dataset = ImageNetHF(image_size, random_crop=random_crop, random_flip=random_flip, split='train', classes=class_cond)
    else:
        dataset = ImageNetHF(image_size, random_crop=random_crop, random_flip=random_flip, split='val', classes=class_cond)
    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader # put all items of loader into list and concat all list infinitely



def load_dataset_MNIST(*,
                        train=True,
                        batch_size,
                        class_cond=True):

    os.makedirs("./data", exist_ok=True)
    if train:
        dataset = MNISTHF(classes=class_cond, split='train')
    else:
        dataset = MNISTHF(classes=class_cond, split='val')
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader

def load_dataset_CIFAR10(*,
                        train=True,
                        batch_size,
                        class_cond=True, data_folder=None):

    os.makedirs("./data", exist_ok=True)
    if train:
        dataset = CIFAR10HF(local_classes=class_cond, split='train', data_folder=data_folder)
    else:
        dataset = CIFAR10HF(local_classes=class_cond, split='val', data_folder=data_folder)
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader

def load_dataset_MNIST_nosampler(*,
                        train=True,
                        batch_size,
                        class_cond=True):

    os.makedirs("./data", exist_ok=True)
    if train:
        dataset = MNISTHF(classes=class_cond, split='train')
    else:
        dataset = MNISTHF(classes=class_cond, split='val')
    # data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size, num_workers=8, shuffle=True)
    while True:
        yield from loader

def load_dataset_CelebA_nosampler(*,
                                  train=True,
                                  batch_size,
                                  class_cond=True,
                                  random_crop=False,
                                  random_flip=True,
                                  image_size):

    os.makedirs("./data", exist_ok=True)
    if train:
        dataset = CelebALocal(root="../data_local/", classes=class_cond, split='train', resolution=image_size,
                           random_crop=random_crop, random_flip=random_flip)
    else:
        dataset = CelebALocal(root="../data_local/", classes=class_cond, split='val', resolution=image_size,
                           random_crop=random_crop, random_flip=random_flip)
    # data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size, num_workers=8, shuffle=True)
    while True:
        yield from loader


def load_dataset_CelebA(*,
                        train=True,
                        batch_size,
                        class_cond=True,
                        random_crop=False,
                        random_flip=True,
                        image_size):
    os.makedirs("./data", exist_ok=True)
    if train:
        dataset = CelebAHF(root="./data/", classes=class_cond, split='train', resolution=image_size,
                           random_crop=random_crop, random_flip=random_flip)
    else:
        dataset = CelebAHF(root="./data/", classes=class_cond, split='val', resolution=image_size,
                           random_crop=random_crop, random_flip=random_flip)
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader

def load_dataset_CelebAUP(*,
                        train=True,
                        batch_size,
                        class_cond=True,
                        random_crop=False,
                        random_flip=True,
                        image_size):
    # get two instances 1 32 1 image_size
    os.makedirs("./data", exist_ok=True)
    if train:
        dataset = CelebA32UPHF(root="./data/", classes=class_cond, split='train', resolution=image_size,
                           random_crop=random_crop, random_flip=random_flip)
    else:
        dataset = CelebA32UPHF(root="./data/", classes=class_cond, split='val', resolution=image_size,
                           random_crop=random_crop, random_flip=random_flip)
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader

def load_dataset_CelebA64(*,
                        train=True,
                        batch_size,
                        class_cond=True,
                        random_crop=False,
                        random_flip=True,
                        image_size):
    os.makedirs("./data", exist_ok=True)
    if train:
        dataset = CelebA64HF(root="./data/", classes=class_cond, split='train', resolution=image_size,
                           random_crop=random_crop, random_flip=random_flip)
    else:
        dataset = CelebA64HF(root="./data/", classes=class_cond, split='val', resolution=image_size,
                           random_crop=random_crop, random_flip=random_flip)
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True)
    while True:
        yield from loader



def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results



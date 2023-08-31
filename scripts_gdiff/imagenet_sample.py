"""
Train a noised image classifier on ImageNet.
"""
# import hfai_env
# hfai_env.set_env('dbg')
import argparse
import os
import datetime

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from hfai.nn.parallel import DistributedDataParallel as DDP
import hfai
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data_imagenet_hfai
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
    model_and_diffusion_defaults,
    classifier_defaults,
)
import numpy as np
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def main(local_rank):
    args = create_argparser().parse_args()
    save_model_folder = os.path.join(args.logdir, "reference")
    os.makedirs(save_model_folder, exist_ok=True)
    dist_util.setup_dist(local_rank)

    log_folder = os.path.join(
        args.logdir,
        datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
    )
    if dist.get_rank() == 0:
        logger.configure(log_folder, rank=dist.get_rank())
    else:
        logger.configure(rank=dist.get_rank())

    logger.log("Loading data")

    data = load_data_imagenet_hfai(train=True, image_size=args.image_size,
                                   batch_size=args.batch_size, random_crop=True)
    if args.val_data_dir:
        # val_data = load_data(
        #     data_dir=args.val_data_dir,
        #     batch_size=args.batch_size,
        #     image_size=args.image_size,
        #     class_cond=True,
        # )
        val_data = load_data_imagenet_hfai(train=False, image_size=args.image_size,
                                           batch_size=args.batch_size, random_crop=False)
    else:
        val_data = None


    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        batch, extra = next(data)
        labels = extra['y']
        gathered_samples = [th.zeros_like(batch) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, batch)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(labels) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, labels)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(save_model_folder, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, model_folder="runs", latest=False):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(model_folder, f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(model_folder, f"opt{step:06d}.pt"))

def save_model_latest(mp_trainer, opt, step, model_folder="runs"):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(model_folder, "latest.pt"),
        )
        th.save({'opt': opt.state_dict(),
                 'step': step}, os.path.join(model_folder, "optlatest.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)

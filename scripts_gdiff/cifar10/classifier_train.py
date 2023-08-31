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
# import torch.distributed as dist
import hfai.nccl.distributed as dist
import torch.nn.functional as F
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from hfai.nn.parallel import DistributedDataParallel as DDP
import hfai
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_dataset_CIFAR10
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
)
from guided_diffusion.script_util_32 import create_classifier_and_diffusion_cifar10

from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
import hfai.client
import hfai.multiprocessing


def main(local_rank):
    args = create_argparser().parse_args()
    save_model_folder = os.path.join(args.logdir, "models")
    os.makedirs(save_model_folder, exist_ok=True)
    dist_util.setup_dist(local_rank)

    log_folder = os.path.join(
        args.logdir,
        "logs",
    )
    if dist.get_rank() == 0:
        logger.configure(log_folder, rank=dist.get_rank())
    else:
        logger.configure(rank=dist.get_rank())
    logger.log("creating model and diffusion...")

    model, diffusion = create_classifier_and_diffusion_cifar10(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0

    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)

        logger.log(
            f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
        )
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint
            )
        )
        load_last_checkpoint=False
    else:
        logger.log(
            f"checking latest.pt model exist at: {save_model_folder}"
        )
        latest_model = os.path.join(save_model_folder, "latest.pt")
        if not os.path.isfile(latest_model):
            logger.log(
                "No latest checkpoint found - train from scratch"
            )
            load_last_checkpoint = False
        else:
            load_last_checkpoint = True
            logger.log(
                "latest Checkpoint found, loading.. "
            )
            model.load_state_dict(dist_util.load_state_dict(latest_model))





    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        # output_device=dist_util.dev(),
        broadcast_buffers=False,
        # bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=True,
    #     random_crop=True,
    # )
    data = load_dataset_CIFAR10(train=True, batch_size=args.batch_size, class_cond=True, data_folder=None)
    if args.val_data_dir:
        # val_data = load_data(
        #     data_dir=args.val_data_dir,
        #     batch_size=args.batch_size,
        #     image_size=args.image_size,
        #     class_cond=True,
        # )
        val_data = load_dataset_CIFAR10(train=True, batch_size=args.batch_size, class_cond=True, data_folder=None)
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint)
        )
    else:
        if load_last_checkpoint:
            opt_checkpoint = os.path.join(save_model_folder, "optlatest.pt")
            if os.path.isfile(opt_checkpoint):
                logger.log(f"Loading optimizer state from checkpoint: {opt_checkpoint}")
                opt_dict = dist_util.load_state_dict(opt_checkpoint)
                opt.load_state_dict(opt_dict["opt"])
                step = opt_dict['step'] + 1
                logger.log(f"Training from {step}")
                resume_step = step



    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
    data_iter = iter(data)
    if val_data is not None:
        val_iter = iter(val_data)

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data_iter)
        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_iter, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step, save_model_folder)
        if step % 1000 == 0 and dist.get_rank() == 0 and step != 0:
            logger.log("Saving latest model")
            save_model_latest(mp_trainer, opt, step+resume_step, save_model_folder)
        elif dist.get_rank() == 0 and hfai.client.receive_suspend_command():
            if step != 0:
                logger.log("Saving latest model")
                save_model_latest(mp_trainer, opt, step + resume_step, save_model_folder)
                logger.log(f"step {step + resume_step}, client has suspended. Good luck next run ^^")
            else:
                logger.log(f"Do not save latest model due to the risk from the first iteration at step {step}")
                logger.log(f"step {step + resume_step - 1}, client has suspended. Good luck next run ^^")
            hfai.client.go_suspend()

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step, save_model_folder)

    dist.barrier()


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
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=500,
        eval_interval=5,
        save_interval=25000,
        logdir="runs"
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)

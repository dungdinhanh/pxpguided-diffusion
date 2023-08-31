import copy
import functools
import glob
import os

import blobfile as bf
import hfai.client
import torch as th
# import torch.distributed as dist
import hfai.nccl.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from hfai.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
# from hfai.checkpoint import load
from guided_diffusion.hfai_checkpoint_support import save_state, load
import time

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        logdir="runs"
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.logdir = os.path.join(logdir, "models")
        os.makedirs(self.logdir, exist_ok=True)
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            if self.load_last_checkpoint:
                self._load_optimizer_state()
                # Model was resumed, either due to a restart or a checkpoint
                # being specified at the command line.
                self.ema_params = [
                    self._load_ema_parameters(rate) for rate in self.ema_rate
                ]
            else:
                self.ema_params = [
                    copy.deepcopy(self.mp_trainer.master_params)
                    for _ in range(len(self.ema_rate))
                ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                # output_device=dist_util.dev(),
                broadcast_buffers=False,
                # bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint) + 1
            if dist.get_rank() == 0:
                print(f"Training from {self.resume_step}")
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
            self.load_last_checkpoint = False
            # self.state_load_state_dict = load(resume_checkpoint, map_location='cpu')
            # logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            # self.model.load_state_dict(self.state_load_state_dict['model'])
            # self.load_last_checkpoint = False
        else:
            last_checkpoint = find_resume_checkpoint(self.logdir)
            if last_checkpoint is not None:
                self.state_load_state_dict = load(last_checkpoint, map_location='cpu')
                logger.log(f"Loading model from latest checkpoint: {last_checkpoint}")
                self.model.load_state_dict(self.state_load_state_dict['model'])
                self.load_last_checkpoint = True
            else:
                self.load_last_checkpoint = False
                logger.log(f"Do not load any checkpoint/train from scratch!")
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        else:
            if self.load_last_checkpoint:
                ema_latest = find_last_ema_checkpoint(self.logdir, rate)
                if ema_latest is None:
                    logger.log(f"No latest ema checkpoint found - Exiting")
                    exit(0)
                logger.log(f"Loading EMA from latest checkpoint: ${ema_latest}...")
                state_dict = load(ema_latest, map_location='cpu')['model']
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        for i in range(len(ema_params)):
            ema_params[i] = ema_params[i].to(dist_util.dev())
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            if self.load_last_checkpoint:
                logger.log(f"Loading optimizer state from latest checkpoint")
                self.opt.load_state_dict(self.state_load_state_dict['optimizer'])
                self.resume_step = self.state_load_state_dict['step'] + 1
                logger.log(f"Train from {self.resume_step}")
        self.state_load_state_dict = None


    def run_loop(self):
        if self.suspend_signal():
            hfai.client.go_suspend()
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if (self.step + self.resume_step) % self.save_interval == 0 and self.step > 0:
                self.save_final()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step % 1000 == 0 and self.step != 0:
                self.save(latest=True)
            elif self.suspend_signal():
                self.save(latest=True)
                logger.log(f"step {self.step + self.resume_step}, client has suspended. Good luck next run ^^")
                if dist.get_rank() == 0:
                    hfai.client.go_suspend()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save_final()

    def suspend_signal(self):
        receive_suspend = hfai.client.receive_suspend_command()
        signal = th.tensor(receive_suspend).bool().cuda()
        dist.broadcast(signal, src=0)
        receive_suspend = signal.item()
        return receive_suspend

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, latest=False):
        def save_checkpoint(rate, params, optimizer):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            save_file_path = os.path.join(self.logdir, filename)
            save_state(save_file_path, state_dict, self.mp_trainer.model, optimizer, others={'step': self.step + self.resume_step})

        def save_latest(rate, params, optimizer):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"latest.pt"
            else:
                filename = f"ema_{rate}_latest.pt"
                optimizer = None
            save_file_path = os.path.join(self.logdir, filename)
            save_state(save_file_path, state_dict, self.mp_trainer.model, optimizer, others={'step': self.step + self.resume_step})
        if not latest:
            save_checkpoint(0, self.mp_trainer.master_params, self.opt)
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params, None)
        else:
            save_latest(0, self.mp_trainer.master_params, self.opt)
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_latest(rate, params, None)
        # remove_temp_mark(self.logdir)
        dist.barrier()

    def save_final(self, latest=False):
        def save_checkpoint_final(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}_final.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}_final.pt"
                with bf.BlobFile(bf.join(self.logdir, filename), "wb") as f:
                    th.save(state_dict, f)

        def save_latest_final(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"final.pt"
                else:
                    filename = f"ema_{rate}_final.pt"
                with bf.BlobFile(bf.join(self.logdir, filename), "wb") as f:
                    th.save(state_dict, f)
        if not latest:
            save_checkpoint_final(0, self.mp_trainer.master_params)
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint_final(rate, params)

            if dist.get_rank() == 0:
                with bf.BlobFile(
                    bf.join(self.logdir, f"opt{(self.step + self.resume_step):06d}_final.pt"),
                    "wb",
                ) as f:
                    th.save(self.opt.state_dict(), f)
        else:
            save_latest_final(0, self.mp_trainer.master_params)
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_latest_final(rate, params)

            if dist.get_rank() == 0:
                with bf.BlobFile(
                        bf.join(self.logdir, f"optlatest_final.pt"),
                        "wb",
                ) as f:
                    th.save({'opt': self.opt.state_dict(),
                             'step': self.step + self.resume_step}, f)

        dist.barrier()


def remove_temp_mark(save_folder):
    list_temp_files = list(glob.glob(os.path.join(save_folder, "*_temp.pt")))
    list_new_files = remove_temp_mark_str(list_temp_files)
    for i in range(len(list_new_files)):
        os.rename(list_temp_files[i], list_new_files[i])


def remove_temp_mark_str(list_files):
    new_list_files = []
    for file in list_files:
        new_file = file[:-8] + ".pt"
        new_list_files.append(new_file)
    return new_list_files
    pass


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    split1 = split1.split("_")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint(logdir):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    latest_checkpoint = os.path.join(logdir, "latest.pt")
    if os.path.isdir(latest_checkpoint) or os.path.isfile(latest_checkpoint):
        return latest_checkpoint
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def find_last_ema_checkpoint(logdir, rate):
    if logdir is None:
        return None
    filename = f"ema_{rate}_latest.pt"
    path = bf.join(logdir, filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def model_entropy(x_pred):
    # compute entropy loss
    x_pred = th.mean(x_pred, dim=0)
    loss = x_pred * th.log(x_pred + 1e-20)
    return th.sum(loss)

# if __name__ == '__main__':
#     list_files = ["home/abc//runs/latest_temp.opt", "home/abc//runs/ema_0.9999_latest_temp.opt"]
#     new_files = remove_temp_mark_str(list_files)
#     print(new_files)
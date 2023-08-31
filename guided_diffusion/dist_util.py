"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
# from mpi4py import MPI
import torch as th
# import torch.distributed as dist
import hfai.nccl.distributed as dist
import torch.cuda

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(local_rank):
    """
    Set up a distributed process group.
    """
    if dist.is_initialized():
        return
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # 机器个数 number of machines note
    rank = int(os.environ['RANK'])  # 当前机器编号 the rank of current note
    gpus = th.cuda.device_count()  # 每台机器的GPU个数 the number of GPU on each GPU

    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts * gpus,
                            rank=rank * gpus + local_rank)

    th.cuda.set_device(local_rank)
    # comm = MPI.COMM_WORLD
    # backend = "gloo" if not th.cuda.is_available() else "nccl"
    #
    # if backend == "gloo":
    #     hostname = "localhost"
    # else:
    #     hostname = socket.gethostbyname(socket.getfqdn())
    # os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    # os.environ["RANK"] = str(comm.rank)
    # os.environ["WORLD_SIZE"] = str(comm.size)
    #
    # port = comm.bcast(_find_free_port(), root=0)
    # os.environ["MASTER_PORT"] = str(port)
    # dist.init_process_group(backend=backend, init_method="env://")



def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load PyTorch file
    """

    return th.load(path, map_location='cpu')


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

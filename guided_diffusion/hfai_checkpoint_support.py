from pathlib import Path
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.optim import Optimizer

from hfai.checkpoint.utils import check_type
from hfai.checkpoint.dist_ckpt import OptimizerSharder

from guided_diffusion.fp16_util import MixedPrecisionTrainer
import heapq
import warnings
from collections import OrderedDict

LARGE_SIZE = 256 * (1 << 20)  # 256 MB
VERSION = "2.0.0"


def save_state(fname, model_params, model, optimizer, others, group=None) -> None:
    """
    该函数把 checkpoint 切分成多份，每个 rank 保存一份数据，从而加快保存 checkpoint 的速度。

    Args:
        fname (str, os.PathLike): 保存的文件位置
        model (state_dict, List[state_dict]): state_dict or list of state_dict
        optimizer (Optimizer, List[Optimizer]): 优化器，可以是包含多个优化器对象的 ``list``，如果是 None 则忽略，默认是 ``None``
        others (dict): 其他需要保存的一些信息，默认是 ``None``
        group (ProcessGroup): ProcessGroup 对象，默认是 ``None``

    Examples:

    .. code-block:: python

        from hfai.checkpoint import save, load

        model = DistributedDataParallel(model, device_ids=[local_rank])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # training ...
        for epoch in range(epochs):
            for step, data in enumerate(dataloader):
                # training

                others = {'epoch': epoch, 'step': step + 1}

                if receive_suspend:
                    save('latest.pt', model, optimizer, others=others)

        # 恢复训练
        state = load('latest.pt', map_location='cpu')
        epoch, step = state['epoch'], state['step']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    NOTE:
        模型的 buffer 只会保存 rank-0 的

    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized yet.")

    if others is not None and not isinstance(others, dict):
        raise TypeError(f"`others` could only be None or a dict")

    assert group is None or isinstance(group, dist.ProcessGroup)

    model_params = check_type("model", model_params, dict)
    optimizers = check_type("optimizer", optimizer, Optimizer)

    others = others or {}
    for n in others:
        assert n not in ["model", "optimizer"], n

    group = group or dist.distributed_c10d._get_default_group()
    rank = dist.get_rank(group=group)
    nshards = dist.get_world_size(group=group)
    state = {"__nshards__": nshards, "__version__": VERSION}

    sharder = ModelStateSharder(group)
    state['model'] = [sharder.apply(model_param, model) for model_param in model_params]

    sharder = OptimizerSharderCustom(group)
    state['optimizer'] = [sharder.apply(opt) for opt in optimizers]
    # save others by rank-0
    if rank == 0:
        state.update(others)

    # write to the filesystem
    output_dir = Path(fname)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / f"PART_{rank:03d}.pt")
    dist.barrier(group=group)


import numpy as np


class ModelStateSharder():

    def __init__(self, group):
        self.rank = dist.get_rank(group=group)
        self.nshards = dist.get_world_size(group=group)

    def apply(self, model_params, model):
        params = model_params

        buffers = {name: params.pop(name) for name, buf in model.named_buffers()}
        large_params = self.collect_large_params(params)
        params = split_dict_model(params, self.rank, self.nshards)

        if self.rank == 0:
            params.update(buffers)

        return params, large_params

    def collect_large_params(self, params):
        large_params = {}

        for name in list(params.keys()):
            param = params[name]
            size = param.numel() * param.element_size()
            if size > LARGE_SIZE and param.layout == torch.strided:
                # 这里得用 clone(), 否则实际上会存整个 tensor
                new_p = param.view(-1).chunk(self.nshards)[self.rank].clone()
                large_params[name] = (param.shape, new_p)
                del params[name]

        return large_params


import copy


class OptimizerSharderCustom():

    def __init__(self, group):
        self.rank = dist.get_rank(group=group)
        self.nshards = dist.get_world_size(group=group)

    def apply(self, optimizer):
        opt = copy.deepcopy(optimizer.state_dict())
        state = opt["state"]
        large_tensors = self.collect_large_tensors(state)
        opt["state"] = split_dict(state, self.rank, self.nshards)
        return opt, large_tensors

    def collect_large_tensors(self, state):
        large_tensors = {}

        for k1 in list(state.keys()):
            if not isinstance(state[k1], dict):
                continue

            for k2 in list(state[k1].keys()):
                x = state[k1][k2]
                if not isinstance(x, torch.Tensor):
                    continue

                size = x.numel() * x.element_size()
                if size > LARGE_SIZE and x.layout == torch.strided:
                    # 这里得用 clone(), 否则实际上会存整个 tensor
                    new_x = x.view(-1).chunk(self.nshards)[self.rank].clone()
                    large_tensors[(k1, k2)] = (x.shape, new_x)
                    del state[k1][k2]

        return large_tensors


def load(fname, nthreads=8, **kwargs) -> dict:
    """
    加载通过 `hfai.checkpoint.save` 保存的 checkpoint

    Args:
        fname (str, os.PathLike): 保存的文件位置
        nthreads (int): 读取 checkpoint 的线程数，默认是 ``8``
        **kwargs: 传给 ``torch.load`` 的参数

    Returns:
        state (dict): 加载上来的 checkpoint

    Examples:

    .. code-block:: python

        from hfai.checkpoint import save, load

        others = {'epoch': epoch, 'step': step + 1}
        save('latest.pt', model, optimizer, others=others)

        # 恢复训练
        state = load('latest.pt', map_location='cpu')
        epoch, step = state['epoch'], state['step']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    """
    if nthreads < 0:
        raise ValueError(f"nthreads must be >= 0, but found {nthreads}")

    ckpt_dir = Path(fname)
    assert ckpt_dir.is_dir()
    nshards = len(list(ckpt_dir.glob('PART_*.pt')))
    states = [None for _ in range(nshards)]

    # read
    if nthreads > 0:
        def work(i):
            states[i] = torch.load(ckpt_dir / f"PART_{i:03d}.pt", **kwargs)

        with ThreadPool(nthreads) as pool:
            pool.map(work, range(nshards))

        state0 = states[0]
    else:
        state0 = torch.load(ckpt_dir / f"PART_000.pt", **kwargs)

    version = state0.get("__version__", "1.0.0")
    assert version == VERSION

    assert nshards == state0["__nshards__"]
    models = [ModelLoaderCustom() for _ in range(len(state0['model']))]
    opts = [OptimizerLoaderCustom() for _ in range(len(state0['optimizer']))]

    # concat
    for i in range(0, nshards):
        if i == 0:
            state = state0
        else:
            if nthreads > 1:
                state = states[i]
                states[i] = None
            else:
                state = torch.load(ckpt_dir / f"PART_{i:03d}.pt", **kwargs)

        assert nshards == state["__nshards__"]
        for s, model in zip(state['model'], models):
            model.append(*s)
        for s, opt in zip(state['optimizer'], opts):
            opt.append(*s)

    models = [model.finalize() for model in models]
    opts = [opt.finalize() for opt in opts]

    if len(models) == 1:
        models = models[0]
    if len(opts) == 1:
        opts = opts[0]
    if len(opts) == 0:
        opts = None

    state0['model'] = models
    if opts is not None:
        state0['optimizer'] = opts

    return state0


class ModelLoaderCustom():
    """
    Helper class for loading sharded model checkpoint
    """

    def __init__(self):
        self.large_params = defaultdict(list)
        self.large_params_shape = {}
        self.params = {}

    def append(self, params, large_params):
        self.params.update(params)
        for name in large_params:
            shape, param = large_params[name]
            self.large_params[name].append(param)
            self.large_params_shape[name] = shape

    def finalize(self):
        for name in list(self.large_params.keys()):
            assert name not in self.params
            param = self.large_params[name]
            shape = self.large_params_shape[name]
            param = torch.cat(param, dim=0).view(*shape)
            self.params[name] = param

            del self.large_params[name]

        params = self.params
        self.params = None
        self.large_params = None

        return params


class OptimizerLoaderCustom():

    def __init__(self):
        self.large_tensors = defaultdict(list)
        self.large_tensors_shape = {}
        self.opt = {}

    def append(self, opt, large_tensors):
        if len(self.opt) == 0:
            self.opt = opt
        else:
            self.opt["state"].update(opt["state"])

        for key in large_tensors:
            shape, tensor = large_tensors[key]
            self.large_tensors[key].append(tensor)
            self.large_tensors_shape[key] = shape

    def finalize(self):
        for key in list(self.large_tensors.keys()):
            k1, k2 = key
            assert k2 not in self.opt["state"][k1]

            tensors = self.large_tensors[key]
            shape = self.large_tensors_shape[key]
            tensor = torch.cat(tensors, dim=0).view(*shape)
            self.opt["state"][k1][k2] = tensor
            del self.large_tensors[key]

        opt = self.opt
        self.opt = None
        self.large_tensors = None

        return opt

def split_dict_model(state, rank, nshards):
    keys = list(state.keys())
    sizes = [count_size(state[k]) for k in keys]

    # 计算每个 rank 应该负责哪部分的 checkpoint
    sets = divide_almost_equally(sizes, nshards)

    new_state = OrderedDict()
    for i in sets[rank]:
        k = keys[i]
        new_state[k] = state[k].clone()

    return new_state


def split_dict(state, rank, nshards):
    keys = list(state.keys())
    sizes = [count_size(state[k]) for k in keys]

    # 计算每个 rank 应该负责哪部分的 checkpoint
    sets = divide_almost_equally(sizes, nshards)

    new_state = OrderedDict()
    for i in sets[rank]:
        k = keys[i]
        new_state[k] = state[k]

    return new_state


def divide_almost_equally(arr, num_chunks):
    # modified from https://stackoverflow.com/questions/63390126
    arr = np.array(arr)
    indices = np.argsort(arr)[::-1]  # in reverse order
    arr = arr[indices]

    heap = [(0, idx) for idx in range(num_chunks)]
    heapq.heapify(heap)
    sets = OrderedDict()
    for i in range(num_chunks):
        sets[i] = []

    arr_idx = 0
    while arr_idx < len(arr):
        set_sum, set_idx = heapq.heappop(heap)
        sets[set_idx].append(indices[arr_idx])
        set_sum += arr[arr_idx]
        heapq.heappush(heap, (set_sum, set_idx))
        arr_idx += 1

    return list(sets.values())


def count_size(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()

    if isinstance(x, dict):
        return sum([count_size(a) for a in x.values()])

    if isinstance(x, (list, tuple)):
        return sum([count_size(a) for a in x])

    if isinstance(x, (float, int)):
        return 8

    return 0
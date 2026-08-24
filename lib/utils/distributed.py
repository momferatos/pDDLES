# Copyright (c) 2022-2026 Georgios Momferatos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from types import FrameType
from typing import Any, Iterable, Iterator

import torch
import os, random, socket
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from collections import defaultdict, deque
import time, datetime, signal
import subprocess
from pathlib import Path
import submitit
import random


"""
Misc functions.
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""



def setup_for_distributed(is_master: bool) -> None:
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def fix_random_seeds(seed: int = 31) -> None:
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_shared_folder(args: argparse.Namespace) -> Path:
    path = os.environ['pDDLES']
    if Path(path).is_dir():
        p = Path(os.path.join(path, args.prefix))
        p.mkdir(exist_ok=True)
    else:
        raise RuntimeError(
            "No shared folder found. Please set the environment variable "
            "pDDLES to a shared folder path.")
    return p

def init_dist_node(args: argparse.Namespace) -> None:

    if args.slurm:

        args.ngpus_per_node = args.tasks_per_node

        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)

        # distributed parameters
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node

    else:

        if args.dev == 'gpu':
            # One process per GPU listed in -gpus (default '0' => a single
            # GPU). Set CUDA_VISIBLE_DEVICES before any CUDA call so the listed
            # GPUs are remapped to cuda:0..N-1, matching the local index used
            # to assign each process its device.
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
            args.ngpus_per_node = len(args.gpus.split(','))
        else:
            args.ngpus_per_node = args.tasks_per_node

        args.rank = 0
        args.world_size = args.ngpus_per_node

        if not args.port:
            # pick a free port so two local runs on one machine don't
            # collide on a fixed rendezvous port; chosen once here, before
            # mp.spawn, so every child inherits the same value
            with socket.socket() as s:
                s.bind(('localhost', 0))
                args.port = s.getsockname()[1]

        # Set All the Necessary Environment Variables!
        os.environ["MASTER_ADDR"] = 'localhost'
        os.environ["MASTER_PORT"] = str(args.port)
        #os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
        #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def init_dist_gpu(gpu: int | None, args: argparse.Namespace) -> None:

    if args.slurm:
        job_env = submitit.JobEnvironment()
        args.output_dir = Path(str(args.output_dir).replace(
            "%j", str(job_env.job_id)))
        args.gpu = job_env.local_rank
        args.localrank = job_env.local_rank
        args.rank = job_env.global_rank
        nodelist  = os.environ["SLURM_JOB_NODELIST"]

        # 6. MASTER_ADDR
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]

        # 7. MASTER_PORT
        if args.port:
            port = args.port
        else:
            # deterministic per-job port: every rank of a job derives the
            # same value, and two jobs sharing a node get different ports
            jobid = int(''.join(c for c in str(job_env.job_id)
                                if c.isdigit()) or '0')
            port = 40000 + jobid % 20000

        print(f'Process {args.rank}/{args.world_size} @ {host_name}:{port}')

        # Set All the Necessary Environment Variables!
        os.environ["MASTER_ADDR"] = host_name
        os.environ["MASTER_PORT"] = str(port)
#        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
#        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    else:
        args.gpu = gpu
        args.rank += gpu

    if args.dev == 'gpu':
        backend = 'nccl'
        # Bind each rank to its node-local GPU. Under slurm every task sees
        # all of the node's GPUs (gres=gpu:N without gpu-bind), so a fixed
        # cuda:0 would stack all local ranks on one device; args.gpu is the
        # local rank in both modes (JobEnvironment.local_rank / spawn index,
        # remapped via CUDA_VISIBLE_DEVICES for local runs). Passing
        # device_id to init_process_group also stops collectives like
        # barrier() from guessing the device.
        args.device = torch.device('cuda', args.gpu)
        device_id = args.device
        torch.cuda.set_device(device_id)
    else:
        backend = 'gloo'
        args.device = torch.device('cpu')
        device_id = None

    dist.init_process_group(
        backend=backend, world_size=args.world_size, rank=args.rank,
        device_id=device_id)
    #init_method=args.dist_url

    fix_random_seeds()

    cudnn.benchmark = True

    dist.barrier()

    args.main = (args.rank == 0)
    setup_for_distributed(args.main)

def cleanup_distributed() -> None:
    """Tear down the process group on exit.

    Without this, NCCL warns about leaked resources at program exit. The
    barrier makes all ranks finish before any of them destroys the group.
    """
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def handle_sigusr1(signum: int, frame: FrameType | None) -> None:
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum: int, frame: FrameType | None) -> None:
    pass

def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        if fmt is None:
            fmt = "{median:.6e} ({global_avg:.6e})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt


    def update(self, value: float, n: int = 1,
               args: argparse.Namespace | None = None) -> None:
        self.args = args
        self.deque.append(value)
        self.count += n
        self.total += value * n
        if args:
            self.device = args.device


    def synchronize_between_processes(self) -> None:
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device=self.device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, args: argparse.Namespace, delimiter: str = "\t") -> None:
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.args = args

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            # assert isinstance(v, (float, int))
            self.meters[k].update(v, args=self.args)

    def __getattr__(self, attr: str) -> Any:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():

            if isinstance(meter, float):
                string = f'{meter:.5e}'
            else:
                string = str(meter)

            loss_str.append(
                "{}: {}".format(name, string)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable: Iterable[Any], print_freq: int,
                  args: argparse.Namespace,
                  header: str | None = None) -> Iterator[Any]:
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                #'time: {time}',
                #'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                #'time: {time}',
                #'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end, args=self.args)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

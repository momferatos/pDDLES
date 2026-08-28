#!/usr/bin/env python3
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
import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from lib.utils.distributed import init_dist_node, init_dist_gpu, \
    get_shared_folder, cleanup_distributed

import submitit, random, sys
from pathlib import Path
from lib.post_process.post_process import plot_results, plot_FNet, \
    plot_histograms
from lib.scaler.Scaler import get_scaler
from lib.datasets.Sampler import TurbSampler

import h5py
import numpy as np

import math

import pywt

from parse_args import parse_args


class SLURM_Trainer(object):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def __call__(self) -> None:

        init_dist_node(self.args)
        train(None, self.args)


def main() -> None:
    
    args, parser = parse_args()
    
    cmdline = ''
    for arg in sys.argv:
        cmdline = cmdline + ' ' + arg
    args.cmdline = cmdline
    
    if args.dev == 'gpu':
        if not args.tasks_per_node:
            args.tasks_per_node = 4
        if args.partition is None:
            args.partition = 'gpu'
        slurm_additional_parameters = {'gres': f'gpu:{args.tasks_per_node}', 'time': f'{args.timeout}'}#,  'gpu-bind': 'single:1}
    else:
        if not args.tasks_per_node:
            args.tasks_per_node = 16
        if args.partition is None:
            args.partition = 'cpu'
        slurm_additional_parameters = {'time': f'{args.timeout}'}
    
    

    # 3D volumetric data: the helical decomposition takes cross products of
    # 3D wavevectors, so this is fixed, not a user knob.
    args.dimensions = 3

    # velocity ('u', transposed on load) vs passive scalar ('scl'); set by
    # -scalar because the load-time transpose keys on these exact names.
    args.hdf5_key = ('scl' if args.scalar else 'u')
    args.conv = (nn.Conv2d if args.dimensions == 2 else nn.Conv3d)
    args.batchnorm = (nn.BatchNorm2d if args.dimensions == 2 else nn.BatchNorm3d)

    args.output_dir = get_shared_folder(args) / f'{args.model}'
    args.out = args.output_dir
    args.numpy_dtype = (np.float32 if args.precision == 'single' else np.float64)
    args.torch_dtype = (torch.float32 if args.precision == 'single' else torch.float64)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    
    
    if args.slurm:
        executor = submitit.AutoExecutor(folder=args.output_dir,
                                         slurm_max_num_timeout=30)
        executor.update_parameters(
            mem_gb=args.mem_per_node,
            tasks_per_node=args.tasks_per_node,
            cpus_per_task=1,
            nodes=args.nnodes,
            slurm_partition=args.partition,
            slurm_account=args.account,
            slurm_qos=args.qos,
            slurm_mem=args.mem,
            slurm_additional_parameters=slurm_additional_parameters
        )

        if args.nodelist:
            executor.update_parameters(
                slurm_additional_parameters = {
                    "nodelist": f'{args.nodelist}' })

        executor.update_parameters(name=args.model)
        trainer = SLURM_Trainer(args)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id}")
    else:
        init_dist_node(args)
        mp.spawn(train, args = (args,), nprocs = args.ngpus_per_node)
	

def train(gpu: int | None, args: argparse.Namespace) -> None:

    print()
    print(f'Full command line: {args.cmdline}')
    print()
    
    # === SET ENV === #

    # args.device is assigned inside init_dist_gpu, once the rank's local
    # index is known: submitit's JobEnvironment for slurm, the spawn index
    # for local runs. A fixed device here would stack all of a node's ranks
    # on cuda:0.
    init_dist_gpu(gpu, args)
          
    ngpus = torch.cuda.device_count()
    print(f'Found {ngpus} visible GPU(s):')
    for i in range(ngpus):
        print(f'{i} -  {torch.cuda.get_device_properties(i).name}')
    print()

    print(f'Local rank {args.gpu} is using device {args.device}')
    
    # dataset loader #
    get_dataset = getattr(__import__("lib.datasets.{}".format(args.dataset),
                                     fromlist=["get_dataset"]), "get_dataset")
    
    # read the list of training/test filenames
    with open(args.datafile, 'r') as f:
        filenames = [os.path.abspath(fn.strip()) for fn in list(f)]

    # determine the size of the DNS square/cube, N
    with h5py.File(filenames[0], 'r') as h5file:
        keys = h5file.keys()
        y = np.array(h5file[args.hdf5_key], dtype=args.numpy_dtype)
        args.n = y.shape[0]

    if args.wavelet == 'list':
        for wavelet in pywt.wavelist(kind='discrete'):
            wv = pywt.Wavelet(wavelet)
            levels = pywt.dwt_max_level(args.n, wv)
            if levels > 0:
                print(f'{wavelet} - {levels}/{int(np.log2(args.n))}')
        return
    
    if args.num_coeffs == 0:
        args.num_coeffs = int(math.sqrt(args.n ** 3))
        
    num_files = len(filenames)
    ntrain_test = int(0.8 * num_files)
    ntrain = int(0.6 * num_files)
    g = torch.Generator()
    args.seed = 777
    g.manual_seed(args.seed)
    indices = torch.randperm(num_files, generator=g).tolist()  
    train_filenames = []
    for i in range(ntrain):
        train_filenames.append(filenames[indices[i]])

    test_filenames = []
    for i in range(ntrain, ntrain_test):
        test_filenames.append(filenames[indices[i]])

    valid_filenames = []
    for i in range(ntrain_test, num_files):
        valid_filenames.append(filenames[indices[i]])
        
    train_dataset = get_dataset(train_filenames, args)
    test_dataset = get_dataset(test_filenames, args)
    valid_dataset = get_dataset(valid_filenames, args)

   
    # train_sampler = DistributedSampler(train_dataset,
    #                                    shuffle=args.shuffle,
    #                                    num_replicas = args.world_size,
    #                                    rank = args.rank, seed = 31)
    # test_sampler = DistributedSampler(test_dataset,
    #                                   shuffle=args.shuffle,
    #                                   num_replicas = args.world_size,
    #                                   rank = args.rank, seed = 31)

    train_sampler = TurbSampler(train_dataset,
                                shuffle=args.shuffle,
                                num_replicas=args.world_size,
                                rank=args.rank,
                                seed=31,
                                drop_last=args.drop_last)
    test_sampler = TurbSampler(test_dataset,
                               shuffle=False,
                               num_replicas=args.world_size,
                               rank=args.rank,
                               seed=31,
                               drop_last=args.drop_last)


    train_loader = DataLoader(dataset=train_dataset,
                            sampler=train_sampler,
                            batch_size=args.batch_per_task, 
                            num_workers= args.workers,
                            pin_memory=True,
                            drop_last=args.drop_last)

    test_loader = DataLoader(dataset=test_dataset, 
                            sampler=test_sampler,
                            batch_size=args.batch_per_task, 
                            num_workers=args.workers,
                            pin_memory=True,
                            drop_last=False,
                            shuffle=False)

    # sampler-less: TurbDataset.__len__ is the true file count, so this walks
    # the whole validation set on every rank. The per-rank validation loss is
    # still all-reduced to the same full-set mean, and rank 0 (which does the
    # plotting) now sees all validation files instead of a 1/world_size shard.
    valid_loader = DataLoader(dataset=valid_dataset,
                                sampler=None,
                                batch_size=args.batch_per_task,
                                num_workers=args.workers,
                                pin_memory=True,
                                drop_last=False,
                                shuffle=False)

    scaler_loader = DataLoader(dataset=train_dataset,
                                sampler=None,
                                batch_size=args.batch_per_task,
                                num_workers=args.workers,
                                pin_memory=True,
                                drop_last=False,
                                shuffle=False)

    print(f"Data loaded")
        
    # === MODEL === #
    get_model = getattr(__import__("lib.arch.{}".format(args.arch),
                                   fromlist=["get_model"]), "get_model")
    model = get_model(args)
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    
    print()
    print((f'{args.arch} trainable parameters: {trainable_params} '
           f'in  {args.num_blocks} blocks.'))
    if args.arch == 'FNet':
        print(f'{args.num_coeffs} trainable spectral coefficients per block.')
    elif args.arch == 'WNet':
        print(f'Wavelet: {args.wavelet}')
        if args.num_levels:
            num_levels = args.num_levels
        else:
            w = pywt.Wavelet(args.wavelet)
            num_levels = pywt.dwt_max_level(args.n, w)
        print((f'Number of trainable wavelet levels: '
               f'{num_levels}/{int(np.log2(args.n))}'))
        print()
        
    if args.dev == 'gpu':
        # use if model contains batchnorm.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 

    # DDP only accepts device_ids for single-GPU modules; CPU modules must
    # pass None or DDP raises at construction. args.gpu is this rank's
    # node-local device index in both slurm and local modes.
    if args.dev == 'gpu':
        device_ids = [args.gpu]
    else:
        device_ids = None
        
    model = model.to(args.device)
        
    find_unused_parameters = None
    if args.arch == 'WNet':
        find_unused_parameters = True
        
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters)
    # model = torch.compile(model)
    
    # === LOSS === #
    from lib.core.loss import get_loss
    loss = get_loss(args)

    # === OPTIMIZER === #
    from lib.core.optimizer import get_optimizer
    optimizer = get_optimizer(model, args)

    scaler = get_scaler(scaler_loader, args)
    # Normalization constants are handled by the trainer: load_if_available()
    # fits them on a fresh run or restores them from the checkpoint on resume,
    # and Trainer.save() writes them into every checkpoint (no separate
    # norm.pt file).

    # === TRAINING === #
    Trainer = getattr(__import__("lib.trainers.{}".format(args.trainer),
                                 fromlist=["Trainer"]), "Trainer")
    
    if args.predict:
        Trainer(args,
                train_loader,
                test_loader,
                valid_loader,
                model,
                loss,
                optimizer,
                train_dataset,
                scaler).load_if_available()
        
        print(f"Model loaded")
        print()
        
        train_losses = [[0., 0.], [1., 1.]]
        test_losses = [[0., 0.], [1., 1.]]
    else:

        print(f"Model loaded")
        print()
        
        train_losses, test_losses, valid_loss = Trainer(args,
                                                        train_loader,
                                                        test_loader,
                                                        valid_loader,
                                                        model,
                                                        loss,
                                                        optimizer,
                                                        train_dataset,
                                                        scaler).fit()

        if test_losses:
            min_test_loss = np.min(np.array(test_losses))
            min_epoch = np.argmin(np.array(test_losses))
            min_train_loss = train_losses[min_epoch]
            print((f'Minimum test loss {min_test_loss:.5e} @ epoch {min_epoch}, '
                   f'training loss = {min_train_loss:.5e}, '
                   f'validation loss = {valid_loss:.5e}'))
            with open(os.path.join(args.out, 'losses.dat'), 'w') as f:
                f.write((f'{args.alpha} {min_epoch} {min_test_loss} '
                         f'{min_train_loss} {valid_loss}'))
        else:
            # resumed checkpoint was already at args.epochs: nothing trained
            # this session, so there are no per-epoch losses to summarize
            print(('No epochs trained this session, '
                   f'validation loss = {valid_loss:.5e}'))

    if args.main:
        # rank 0 only: every rank used to redo the same plots and race on
        # the same output files. Plot with the bare module - a DDP forward
        # here would broadcast buffers, a collective the other ranks (already
        # waiting in cleanup_distributed's barrier) never join.
        plot_model = model.module
        plot_histograms(valid_loader, plot_model, train_dataset, scaler, args)
        plot_results(args, plot_model, train_losses, test_losses,
                     train_dataset, valid_loader, scaler)

        if args.arch == 'FNet':
            plot_FNet(plot_model, args)

    cleanup_distributed()

    return


if __name__ == "__main__":
    main()

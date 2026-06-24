#!/usr/bin/env python3
import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from lib.utils.distributed import init_dist_node, init_dist_gpu, \
    get_shared_folder

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
    def __init__(self, args):
        self.args = args

    def __call__(self):

        init_dist_node(self.args)
        train(None, self.args)


def main():
    
    args, parser = parse_args()
    
    cmdline = ''
    for arg in sys.argv:
        cmdline = cmdline + ' ' + arg
    args.cmdline = cmdline
    
    if args.dev == 'gpu':
        if not args.tasks_per_node:
            args.tasks_per_node = 4
        args.partition = 'gpu'
        slurm_additional_parameters = {'gres': f'gpu:{args.tasks_per_node}', 'time': f'{args.timeout}'}#,  'gpu-bind': 'single:1}    
    else:
        if not args.tasks_per_node:
            args.tasks_per_node = 16
        args.partition = 'cpu'
        slurm_additional_parameters = {'time': f'{args.timeout}'}
    
    args.port = 7778
    

    args.dimensions = 3
    
    args.hdf5_key = ('scl' if args.scalar else 'u')
    args.conv = (nn.Conv2d if args.dimensions == 2 else nn.Conv3d)
    args.batchnorm = (nn.BatchNorm2d if args.dimensions == 2 else nn.BatchNorm3d)

    args.output_dir = get_shared_folder(args) / f'{args.model}'
    args.out = args.output_dir
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
	

def train(gpu, args):

    print()
    print(f'Full command line: {args.cmdline}')
    print()
    
    if args.dev == 'gpu':
        if args.slurm:
            args.device = 'cuda:0'
        else:
            # use the local process index, not args.rank: rank is still 0 here
            # (it gets its per-process value later in init_dist_gpu), so using
            # it would put every process on cuda:0.
            args.device = torch.device('cuda', gpu)
    else:
        args.device = torch.device('cpu', args.rank)
            
#    torch.set_default_device(args.device)
    
    # === SET ENV === #
    
    init_dist_gpu(gpu, args)
          
    ngpus = torch.cuda.device_count()
    print(f'Found {ngpus} visible GPU(s):')
    for i in range(ngpus):
        print(f'{i} -  {torch.cuda.get_device_properties(i).name}')
    print()

    print(f'Process {gpu} is using device {args.device}')
    
    # === DATA === #
    get_dataset = getattr(__import__("lib.datasets.{}".format(args.dataset),
                                     fromlist=["get_dataset"]), "get_dataset")
    
    # read the list of training/test filenames
    with open(args.datafile, 'r') as f:
        filenames = [os.path.abspath(fn.strip()) for fn in list(f)]

    args.h5path = os.path.dirname(os.path.realpath(filenames[-1]))
    
    # detrmine the size of the DNS square/cube, N
    with h5py.File(filenames[0], 'r') as h5file:
        keys = h5file.keys()
        y = np.array(h5file[args.hdf5_key], dtype='float32')
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
                               shuffle=args.shuffle,
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
                            drop_last=args.drop_last)

    valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=args.batch_per_task,
                                num_workers=args.workers,
                                pin_memory=True,
                                drop_last=args.drop_last,
                                shuffle=False)

    scaler_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_per_task,
                                num_workers=args.workers,
                                pin_memory=True,
                                drop_last=args.drop_last,
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

    device_ids = None
    if args.slurm:
        if args.dev == 'gpu':
            device_ids = [0]
        else:
            device_ids = None
    else:
        if args.dev == 'gpu':
            device_ids = [gpu]
        else:
            device_ids = [args.device]
        
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
    
    if not scaler.load(args):
        scaler.fit()
    else:
        print('Scaler values loaded from file.')
    scaler.store(args)
    
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

        min_test_loss = np.min(np.array(test_losses))
        min_epoch = np.argmin(np.array(test_losses))
        min_train_loss = train_losses[min_epoch]
        print((f'Minimum test loss {min_test_loss:.5e} @ epoch {min_epoch}, '
               f'training loss = {min_train_loss:.5e}, '
               f'validation loss = {valid_loss:.5e}'))
        with open(os.path.join(args.out, 'losses.dat'), 'w') as f:
            f.write((f'{args.alpha} {min_epoch} {min_test_loss} '
                     f'{min_train_loss} {valid_loss}'))

    plot_histograms(valid_loader, model, train_dataset, scaler, args)
    plot_results(args, model, train_losses, test_losses,
                 train_dataset, valid_loader, scaler)
    
    if args.arch == 'FNet':
        plot_FNet(model, args)
    
    return


if __name__ == "__main__":
    main()

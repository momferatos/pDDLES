# Copyright (c) Ramy Mounir.
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

import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import argparse
from lib.utils.file import bool_flag
from lib.utils.distributed import init_dist_node, init_dist_gpu, get_shared_folder

import submitit, random, sys
from pathlib import Path
from lib.utils.parameters import parameters
from lib.post_process.post_process import plot_results, plot_FourierNet
from lib.datasets.Scaler import get_scaler

import h5py
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description='Template')

    # === PATHS === #

    parser.add_argument('-scalar', type=bool, default=False)
    
    parser.add_argument('-dev', type=str, default="gpu",
                                            help='Device to use')
    
    parser.add_argument('-device', type=str, default="cuda:0",
                                            help='Device to use')
    
    parser.add_argument('-data', type=str, default="data",
                                            help='path to dataset directory')
    parser.add_argument('-out', type=str, default="out",
                                            help='path to out directory')

    # === GENERAL === #
    parser.add_argument('-model', type=str, default="my_model",
                                            help='Model name')
    parser.add_argument('-reset', action='store_true',
                                            help='Reset saved model logs and weights')
    parser.add_argument('-tb', action='store_true',
                                            help='Start TensorBoard')
    parser.add_argument('-gpus', type=str, default="0",
                                            help='GPUs list, only works if not on slurm')
    parser.add_argument('-cfg', type =str,
                                            help='Configuration file')

    # === Dataset === #
    parser.add_argument('-dataset', type=str, default = 'TurbDataset',
                                            help='Dataset to choose')
    parser.add_argument('-batch_per_task', type=int, default = 32,
                                            help='batch size per gpu')
    parser.add_argument('-shuffle', type=bool_flag, default = True,
                                            help='Shuffle dataset')
    parser.add_argument('-workers', type=int, default = 0,
                                            help='number of workers')

    # === Architecture === #
    parser.add_argument('-arch', type=str, default = 'FourierNet',
                                            help='Architecture to choose')

    # === Trainer === #
    parser.add_argument('-trainer', type=str, default = 'trainer',
                                            help='Trainer to choose')
    parser.add_argument('-epochs', type=int, default = 10,
                                            help='number of epochs')
    parser.add_argument('-save_every', type=int, default = 1,
                                            help='Save frequency')
    parser.add_argument('-fp16', action='store_true',
                                            help='Use mixed precision only with RTX, V100 and A100 GPUs')


    # === Optimization === #
    parser.add_argument('-optimizer', type=str, default = 'adam',
                                            help='Optimizer function to choose')
    parser.add_argument('-lr_start', type=float, default = 1e-2,
                                            help='Initial Learning Rate')
    parser.add_argument('-lr_end', type=float, default = 1e-3,
                                            help='Final Learning Rate')
    parser.add_argument('-lr_warmup', type=int, default = 10,
                                            help='warmup epochs for learning rate')

    # === SLURM === #
    parser.add_argument('-slurm', action='store_true',
                                            help='Submit with slurm', default=False)
    parser.add_argument('-tasks_per_node', type=int, default = 4,
                                            help='num of gpus per node')
    parser.add_argument('-nnodes', type=int, default = 1,
                                            help='number of nodes')
    parser.add_argument('-nodelist', default = None,
                                            help='slurm nodeslist. i.e. "GPU17,GPU18"')
    parser.add_argument('-partition', type=str, default = "gpu",
                                            help='slurm partition')
    parser.add_argument('-account', type=str, default = "p200140",
                                            help='slurm account')
    parser.add_argument('-timeout', type=str, default = '2-0:0:0',
                                            help='slurm timeout minimum, reduce if running on the "Quick" partition')
    parser.add_argument('-qos', type=str, default = 'default',
                            help='slurm QoS')
    parser.add_argument('-mem_per_node', type=int, default = 256,
                            help='memory per node')


        
    parser.add_argument('-num_blocks',
                        help='Number of blocks in the ResNet or FourierNet',
                        default=8,
                        type=int)

    parser.add_argument('-num_channels',
                        help='Number of convolutional channels'
                        'in the ResNet or SuperNet',
                        default=8,
                        type=int)

    parser.add_argument('-kernel_size',
                        help='Size of the convolution kernel'
                        'in the ResNet or SuperNet',
                        default=3,
                        type=int)

    parser.add_argument('-dropout',
                        help='Dropout factor'
                        'in the ResNet or SuperNet',
                        default=0.2,
                        type=float)

    parser.add_argument('-num_coeffs',
                        help='Number of trainable spectral coefficients in' 
                        'each FourierBlock',
                        default=128,
                        type=int)

    parser.add_argument('-num_levels',
                        help='Number of Wavelet transform levels ',
                        default=3,
                        type=int)

    parser.add_argument('-dimensions',
                        help='Feature dimensions: 2 for image, 3 for'
                        'volumetric data.',
                        default=3,
                        type=int)

    parser.add_argument('-scaler',
                        help='Data normalization: \'norm\' or \'minmax\'',
                        default='norm',
                        type=str)

    parser.add_argument('-alpha',
                        help='Fraction of the wavenumber range to'
                        'be retained by the low-pass LES filter',
                        default=0.1,
                        type=float)

    parser.add_argument('-datafile',
                        help='Training/testing HDF5 filenames\' list file',
                        default='32.dat',
                        type=str)

    parser.add_argument('-hdf5_key',
                        help='HDF5 dataset key',
                        default='u',
                        type=str)


    parser.add_argument('-actfun',
                        help='Activation function to use:'
                        'Tanh, Sigmoid or ReLU',
                        default='Tanh',
                        type=str)
    
    parser.add_argument('-prediction_mode',
                        help='Prediction mode:'
                        '\'large to all\' or \'large_to_small\'',
                        default='large to all',
                        type=str)
    parser.add_argument('-wavelet_type',
                        help='Pywavelet wavelet type to use.',
                        default='db4',
                        type=str)
    parser.add_argument('-wavelet_mode',
                        help='Wavelet coefficient multiplication mode',
                        default='outer',
                        type=str)

    args = parser.parse_args()

    args.actfun = torch.nn.ReLU()
    
    # === Read CFG File === #
    if args.cfg:
        with open(args.cfg, 'r') as f:
            import ruamel.yaml as yaml
            yml = yaml.safe_load(f)

        # update values from cfg file only if not passed in cmdline
        cmd = [c[1:] for c in sys.argv if c[0]=='-']
        for k,v in yml.items():
            if k not in cmd:
                args.__dict__[k] = v

    return args, parser


class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        init_dist_node(self.args)
        train(None, self.args)


def main():
    
    args, parser = parse_args()
    
    if args.dev == 'gpu':
        args.tasks_per_node = 4
        args.partition = 'gpu'
        slurm_additional_parameters = {'gres': f'gpu:{args.tasks_per_node}', 'gpu-bind': f'single:1', 'time': f'{args.timeout}'}    
    else:
        args.partition = 'cpu'
        slurm_additional_parameters = {'time': f'{args.timeout}'}
    
    args.port = random.randint(49152,65535)
    

    args.dimensions = 3
    
    args.hdf5_key = ('scl' if args.scalar else 'u')
    args.conv = (nn.Conv2d if args.dimensions == 2 else nn.Conv3d)
    args.batchnorm = (nn.BatchNorm2d if args.dimensions == 2 else nn.BatchNorm3d)

    if args.slurm:
        # Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
        args.output_dir = get_shared_folder() / f'{args.model}'
        args.out = args.output_dir
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

        executor.update_parameters(
            mem_gb=args.mem_per_node,
            tasks_per_node=args.tasks_per_node,
            cpus_per_task=1,
            nodes=args.nnodes,
            slurm_partition=args.partition,
            slurm_account=args.account,
            slurm_qos=args.qos,
            slurm_mem='256GB',
            slurm_additional_parameters=slurm_additional_parameters
        )

        if args.nodelist:
            executor.update_parameters(slurm_additional_parameters = {"nodelist": f'{args.nodelist}' })

        executor.update_parameters(name=args.model)
        trainer = SLURM_Trainer(args)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id}")
    else:
        init_dist_node(args)
        mp.spawn(train, args = (args,), nprocs = args.ngpus_per_node)
	

def train(gpu, args):


    if args.dev == 'gpu':
        if args.slurm:
            args.device = 'cuda:0'
        else:
            args.device = f'cuda:{gpu}'
    else:
        args.device = 'cpu'
            
#    torch.set_default_device(args.device)
    
    # === SET ENV === #
    print('initializing environment...')
    init_dist_gpu(gpu, args)
    print('Distributed GPUs initialized.')
    print()
    ngpus = torch.cuda.device_count()
    print(f'Found {ngpus} visible GPU(s):')
    for i in range(ngpus):
        print(f'{i} -  {torch.cuda.get_device_properties(i).name}')
    print()

    print(f'Process {gpu} is using device {args.device}')
    
    # === DATA === #
    get_dataset = getattr(__import__("lib.datasets.{}".format(args.dataset), fromlist=["get_dataset"]), "get_dataset")
    
    # read the list of training/test filenames
    with open(args.datafile, 'r') as f:
        filenames = [os.path.abspath(fn.strip()) for fn in list(f)]
        
    # detrmine the size of the DNS square/cube, N
    with h5py.File(filenames[0], 'r') as h5file:
        keys = h5file.keys()
        y = np.array(h5file[args.hdf5_key], dtype='float32')
        args.n = y.shape[0]
        
    dataset = get_dataset(filenames, args)
    num_files = len(dataset)
    ntrain = int(0.8 * num_files)
    lengths = (ntrain, num_files - ntrain)
    train_dataset, test_dataset = random_split(dataset, lengths)
    train_sampler = DistributedSampler(train_dataset, shuffle=args.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
    test_sampler = DistributedSampler(test_dataset, shuffle=args.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
    train_loader = DataLoader(dataset=train_dataset, 
                            sampler = train_sampler,
                            batch_size=args.batch_per_task, 
                            num_workers= args.workers,
                            pin_memory = True,
                            drop_last = True
                            )
    test_loader = DataLoader(dataset=test_dataset, 
                            sampler = test_sampler,
                            batch_size=args.batch_per_task, 
                            num_workers= args.workers,
                            pin_memory = True,
                            drop_last = True
                            )
    print(f"Data loaded")
    

    #print('Normalization constants:')
    #print(f'X_mean: {dataset.X_mean} | X_std: {dataset.X_std}')
    #print(f'y_mean: {dataset.y_mean} | y_std: {dataset.y_std}')
    
    # === MODEL === #
    get_model = getattr(__import__("lib.arch.{}".format(args.arch), fromlist=["get_model"]), "get_model")
    model = get_model(args)
        
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.

    device_ids = None
    if args.slurm:
        device_ids = [0]
    else:
        device_ids = [gpu]
        
    model = model.to(args.device)
        
    find_unused_parameters = None
    if args.arch == 'WaveletNet':
        find_unused_parameters = True
        
    model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=find_unused_parameters)
    # model = torch.compile(model)
    
    # === LOSS === #
    from lib.core.loss import get_loss
    loss = get_loss(args)

    # === OPTIMIZER === #
    from lib.core.optimizer import get_optimizer
    optimizer = get_optimizer(model, args)

    scaler = get_scaler(train_loader, args)
    scaler.fit()
    
    # === TRAINING === #
    Trainer = getattr(__import__("lib.trainers.{}".format(args.trainer), fromlist=["Trainer"]), "Trainer")
    train_losses, test_losses = Trainer(args, train_loader, test_loader, model, loss, optimizer, dataset, scaler).fit()


    plot_results(args, model, train_losses, test_losses, dataset, test_loader, scaler)
        
    if args.arch == 'FourierNet':
        plot_FourierNet(model, args)
    

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import os
import sys

mode = sys.argv[-1]


num_blocks = [2, 4, 6, 8, 10]
prefix = 'WNET'

def spawn(num_blocks, arch, actfuns):
    for actfun in actfuns:
        for nb in num_blocks:
            modelname = f'{arch}_32_{actfun}_{nb}'
            cmd = f'./pDDLES.py -prefix {prefix} -dev cpu -slurm -timeout 0-10:0:0 -datafile 32.dat -batch_per_task 8 -model {modelname} -epochs 10 -lr_start 1.0e-2 -actfun {actfun} -arch {arch} -num_levels 0 -num_blocks {nb} -wavelet_mode outer'
            print()
            print(f'Model #{nb}/{len(num_blocks)}: {modelname}')
            print(f'Command: {cmd}')
            print(f'Output path: {os.path.join(os.environ["pDDLES"], prefix, modelname)}')
            os.system(cmd)
            print()
    return

def collect(num_blocks, arch, prefixes):
    basepath = os.environ['pDDLES']
    for actfun in actfuns:
        color = (np.random.random(), np.random.random(), np.random.random())
        min_test_losses = []
        min_train_losses = []
        val_losses = []
        for nb in num_blocks:
            path = os.path.join(prefix, f'{arch}_32_{actfun}_{nb}')
            path = os.path.join(basepath, path, 'losses.dat')
            with open(path, 'r') as f:
                string = f.readline()
            items = string.split()
            min_test_losses.append(float(items[2]))
            min_train_losses.append(float(items[3]))
            val_losses.append(float(items[-1]))
        plt.plot(num_blocks, val_losses,  color=color, label=f'{actfun}')
    plt.legend(loc='best')
    plt.show()

actfuns  = ['ReLU', 'Tanh'] #'CELU', 'SELU', 'Sigmoid']
arch = 'WaveletNet'

if mode == '-spawn':
    spawn(num_blocks, arch, actfuns)
else:
    collect(alphas, arch, actfuns)
    

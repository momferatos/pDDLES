#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import os
import sys

mode = sys.argv[-1]

dim = 32
numblocks = 8

arch = 'WNet'

lr = 1.0e-3

timeout = '00-04:00:00'

epochs = 100

def spawn():
    maxlevels = int(np.log2(float(dim)))
    for level in range(1, maxlevels + 1):
        modelname = f'{dim}-{numblocks}_blocks-{level}_levels_outer-lr_-3'
        cmd = f'./pDDLES.py -prefix {arch} -slurm -timeout {timeout} -datafile {dim}.dat -batch_per_task 256 -model {modelname} -arch {arch} -epochs {epochs} -lr_start {lr} -num_blocks {numblocks} -num_levels {level}'
        print()
        print(f'Model #{level}/{maxlevels}: {modelname}')
        print(f'Command: {cmd}')
        print(f'Output path: {os.path.join(os.environ["pDDLES"], "WNet", modelname)}')
        os.system(cmd)
        print()
    return

def collect():
    basepath = os.environ['pDDLES']
    maxlevels = int(np.log2(float(dim)))
    for level in range(1, maxlevels + 1):
        color = (np.random.random(), np.random.random(), np.random.random())
        min_test_losses = []
        min_train_losses = []
        val_losses = []
        for alpha in alphas:
            alpha = round(alpha, 2)
            path = os.path.join(arch, f'{dim}-{numblocks}_blocks-{level}_levels_outer-lr_-3')
            path = os.path.join(basepath, path, 'losses.dat')
            with open(path, 'r') as f:
                string = f.readline()
            items = string.split()
            min_test_losses.append(float(items[2]))
            min_train_losses.append(float(items[3]))
            val_losses.append(float(items[-1]))
        plt.plot(alphas, val_losses,  color=color, label=f'{actfun}')
 #       plt.plot(alphas, min_test_losses, '.-',  color=color, label=f'{prefix}: Test')
 #       plt.plot(alphas, min_train_losses, '--', color=color, label=f'{prefix}: Train')
    plt.legend(loc='best')
    plt.show()


if mode == '-spawn':
    spawn()
else:
    collect(alphas, arch, actfuns)
    

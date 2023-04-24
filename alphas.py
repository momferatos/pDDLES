#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import os
import sys

mode = sys.argv[-1]


alphas = np.linspace(0.05, 0.3, 10)

def spawn(alphas, arch, actfuns):
    for actfun in actfuns:
        for ialpha, alpha in enumerate(alphas):
            alpha = round(alpha, 2)
            modelname = f'{arch}_32_{actfun}_{alpha:.2f}'
            cmd = f'python3 pDDLES.py -prefix {arch} -dev cpu -slurm -timeout 0-2:0:0 -mem 40GB -datafile 32.dat -batch_per_task 8 -model {modelname} -epochs 15 -lr_start 1.0e-3 -actfun {actfun} -alpha {alpha} -arch {arch} -num_blocks 2'
            print()
            print(f'Model #{ialpha + 1}/{len(alphas)}: {modelname}')
            print(f'Command: {cmd}')
            print(f'Output path: {os.path.join(os.environ["pDDLES"], arch, modelname)}')
            os.system(cmd)
            print()
    return

def collect(alphas, arch, prefixes):
    basepath = os.environ['pDDLES']
    for actfun in actfuns:
        color = (np.random.random(), np.random.random(), np.random.random())
        min_test_losses = []
        min_train_losses = []
        val_losses = []
        for alpha in alphas:
            alpha = round(alpha, 2)
            path = os.path.join(arch, f'{arch}_32_{actfun}_{alpha:.2f}')
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

actfuns  = ['ReLU', 'CELU', 'SELU', 'Tanh', 'Sigmoid']
arch = 'ResNet'

if mode == '-spawn':
    spawn(alphas, arch, actfuns)
else:
    collect(alphas, arch, actfuns)
    

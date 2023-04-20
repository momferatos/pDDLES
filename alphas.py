#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import os
import sys

mode = sys.argv[-1]


alphas = np.linspace(0.01, 0.4, 10)

def spawn(alphas, actfun):
    for ialpha, alpha in enumerate(alphas):
        alpha = round(alpha, 2)
        modelname = f'FN32_{actfun}_{alpha:.2f}'
        cmd = f'python3 pDDLES.py -dev cpu -slurm -timeout 0-0:15:0 -mem 40GB -datafile 32.dat -batch_per_task 8 -model {modelname} -epochs 15 -lr_start 1.0e-3 -actfun {actfun} -alpha {alpha}'
        print()
        print(f'Model #{ialpha + 1}/{len(alphas)}: {modelname}')
        print(f'Command: {cmd}')
        print(f'Output path: {os.path.join(os.environ["pDDLES"], modelname)}')
        os.system(cmd)
        print()
    return

def collect(alphas, prefixes):
    basepath = os.environ['pDDLES']
    for prefix in prefixes:
        color = (np.random.random(), np.random.random(), np.random.random())
        min_test_losses = []
        min_train_losses = []
        val_losses = []
        for alpha in alphas:
            alpha = round(alpha, 2)
            path = f'{prefix}_{alpha:.2f}'
            path = os.path.join(basepath, path, 'losses.dat')
            with open(path, 'r') as f:
                string = f.readline()
            items = string.split()
            min_test_losses.append(float(items[2]))
            min_train_losses.append(float(items[3]))
            val_losses.append(float(items[-1]))
        plt.plot(alphas, val_losses,  color=color, label=f'{prefix}: Val.')
 #       plt.plot(alphas, min_test_losses, '.-',  color=color, label=f'{prefix}: Test')
 #       plt.plot(alphas, min_train_losses, '--', color=color, label=f'{prefix}: Train')
    plt.legend(loc='best')
    plt.show()

if mode == '-spawn':
    spawn(alphas, 'PReLU')
else:
    collect(alphas, ['FN32R', 'FN32_CELU', 'FN32_SELU', 'FN32_PReLU'])
    

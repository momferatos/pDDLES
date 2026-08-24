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

def spawn() -> None:
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

def collect() -> None:
    # spawn() varies the number of wavelet levels; read each run's
    # losses.dat and plot the validation loss against the level count.
    basepath = os.environ['pDDLES']
    maxlevels = int(np.log2(float(dim)))
    levels = []
    min_test_losses = []
    min_train_losses = []
    val_losses = []
    for level in range(1, maxlevels + 1):
        path = os.path.join(arch, f'{dim}-{numblocks}_blocks-{level}_levels_outer-lr_-3')
        path = os.path.join(basepath, path, 'losses.dat')
        with open(path, 'r') as f:
            string = f.readline()
        items = string.split()
        min_test_losses.append(float(items[2]))
        min_train_losses.append(float(items[3]))
        val_losses.append(float(items[-1]))
        levels.append(level)
    plt.plot(levels, val_losses, marker='o', label='Validation')
 #  plt.plot(levels, min_test_losses, '.-', label='Test')
 #  plt.plot(levels, min_train_losses, '--', label='Train')
    plt.xlabel('Wavelet levels')
    plt.ylabel('Validation loss')
    plt.legend(loc='best')
    plt.show()


if mode == '-spawn':
    spawn()
else:
    collect()
    

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

import matplotlib.pyplot as plt
import numpy as np
import os

basepath = '/project/home/p200140/pDDLES/FNet'
models = ('FNet128-norm-1',
          'FNet128-norm-2',
          'FNet128-norm-4',
          'FNet128-norm-8-final-final',
          'FNet128-norm-16',
          'FNet128-norm-32-30-epochs-final')
numblocks = np.array([1, 2, 4, 8, 16, 32])

val_losses = []
for model, numblock in zip(models, numblocks):
    path = os.path.join(basepath, model, 'losses.dat')
    with open(path, 'r') as f:
        string = f.readline()
        items = string.split()
        val_losses.append(float(items[-1]))
        
plt.plot(numblocks, np.array(val_losses))
plt.show()

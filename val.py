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

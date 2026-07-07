#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################
import os
import hashlib
import numpy as np
import torch
from lib.datasets.TurbDataset import TurbDataset


def _cache_key(args, filenames):
    """Fingerprint of everything the cached scaler constants depend on.

    The constants are statistics of the LES-filtered training files, so the
    cache must be invalidated when the file list, the filter cutoff, the
    resolution, or the field changes - not just the split seed. Basenames
    (not full paths) so moving the data directory keeps a valid cache.
    """
    names = '\n'.join(os.path.basename(fn) for fn in filenames)
    return {'seed': int(args.seed),
            'alpha': float(args.alpha),
            'n': int(args.n),
            'hdf5_key': str(args.hdf5_key),
            'files': hashlib.sha256(names.encode()).hexdigest()}


def _atomic_save(obj, fname):
    """Write-then-rename so a concurrent reader never sees a partial file."""
    tmp = f'{fname}.tmp.{os.getpid()}'
    torch.save(obj, tmp)
    os.replace(tmp, fname)

class NormScaler(object):
    """Datalolader scaler
       

    Parameters
    ----------
    

    Attributes
    ----------
    
    
    """
    
    def __init__(self, dataloader, args):

        self.dataloader = dataloader
        self.dataset = TurbDataset([], args)
                                   
        self.X_mean = 0.0
        self.X_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0

        self.device = args.device

        return

    def fit(self):
        """Determine normalization constants

        Parameters
        ----------
        train_loder : PyTorch DataLoader
           Training DataLoader

        Returns:
           None

        """

        X_mean = 0.0
        y_mean = 0.0
        fac = 0.0
        nbatches = len(self.dataloader)
        for nbatch, y in enumerate(self.dataloader):
            if nbatch % 10 == 0:
                print(f'Computing mean: {nbatch}/{nbatches}')

            dims = y.shape
            X = self.dataset.LES_filter(y)
            X_mean += torch.sum(X, dim=(0, 2, 3, 4),
                                keepdim=True, dtype=torch.float64)
            y_mean += torch.sum(y, dim=(0, 2, 3, 4),
                                keepdim=True, dtype=torch.float64)
            fac += X.numel() / dims[1]

        X_mean /= fac
        y_mean /= fac

        X_std = 0.0
        y_std = 0.0
        fac = 0.0
        for nbatch, y in enumerate(self.dataloader):
            if nbatch % 10 == 0:
                print(f'Computing std: {nbatch}/{nbatches}')
            dims = y.shape
            X = self.dataset.LES_filter(y)
            tmp = (X - X_mean) ** 2
            X_std += torch.sum(tmp, dim=(0, 2, 3, 4),
                               keepdim=True, dtype=torch.float64)
            tmp = (y - y_mean) ** 2
            y_std += torch.sum(tmp, dim=(0, 2, 3, 4),
                               keepdim=True, dtype=torch.float64)

            fac += X.numel() / dims[1]

        X_std = torch.sqrt(X_std / fac)
        y_std = torch.sqrt(y_std / fac)

        self.X_mean = X_mean.to(torch.float32).to(self.device)
        self.X_std = X_std.to(torch.float32).to(self.device)
        self.y_mean = y_mean.to(torch.float32).to(self.device)
        self.y_std = y_std.to(torch.float32).to(self.device)

        
        return

    def transform(self, X, y, action):
        """Normalize minibatch

        Parameters
        ----------
        X : 4d/5d tensor
           Minibatch

        direction: int
            1 for rescaling to scaled data, -1 for rescaling to unscaled data

        feature : bool
           If true, X is a feature varible,  else X is a target variable

        Returns:
           X : 4d/5d tensor
           Normalized minibatch

        """
        
        rang = self.X_std
        bias = self.X_mean            

        if action == 'scale':
            X_tr = (X - bias) / rang
        elif action == 'unscale':
            X_tr = rang * X + bias
        
        rang = self.y_std
        bias = self.y_mean
       
        if action == 'scale':
            y_tr = (y - bias) / rang
        elif action == 'unscale':
            y_tr = rang * y + bias    


        return X_tr, y_tr

    
    def store(self, args):
        fname = os.path.join(args.h5path, 'norm.pt')
        tens = {'key': _cache_key(args, self.dataloader.dataset.filenames),
                'vals': torch.stack([self.X_mean, self.X_std,
                                     self.y_mean, self.y_std])}
        _atomic_save(tens, fname)

        return

    def load(self, args):
        fname = os.path.join(args.h5path, 'norm.pt')
        if not os.path.isfile(fname):
            return
        tens = torch.load(fname)
        # a mismatch also rejects pre-fingerprint cache files ('key' absent)
        if tens.get('key') != _cache_key(args,
                                         self.dataloader.dataset.filenames):
            print('Scaler cache is stale (data/filter settings changed); '
                  'refitting.')
            return

        self.X_mean = tens['vals'][0].to(torch.float32).to(self.device)
        self.X_std = tens['vals'][1].to(torch.float32).to(self.device)
        self.y_mean = tens['vals'][2].to(torch.float32).to(self.device)
        self.y_std = tens['vals'][3].to(torch.float32).to(self.device)

        return 1
                       
class MinmaxScaler(object):
    """Datalolader scaler
       

    Parameters
    ----------
    

    Attributes
    ----------
    
    
    """
    
    def __init__(self, dataloader, args):

        self.dataloader = dataloader
        self.dataset = TurbDataset([], args)
        self.args = args
        
        self.X_max = 1.0
        self.X_min = 0.0
        self.y_max = 1.0
        self.y_min = 0.0

        self.device = args.device

        return


    def fit(self):
        """Determine normalization constants

        Parameters
        ----------
        train_loder : PyTorch DataLoader
           Training DataLoader

        Returns:
           None

        """

        # calculate maximum/minimum of train dataset for normalization 
        y_min = 1.0e6
        y_max = -1.0e6
        X_min = 1.0e6
        X_max = -1.0e6
        nbatches = len(self.dataloader)
        for nbatch, y in enumerate(self.dataloader):
            if nbatch % 10 == 0:
                print(f'Computing min/max: {nbatch}/{nbatches}')
            X = self.dataset.LES_filter(y)

            X_min = min(X_min, X.flatten().min())
            X_max = max(X_max, X.flatten().max())
            y_min = min(y_min, y.flatten().min())
            y_max = max(y_max, y.flatten().max())
            
        self.X_min = X_min
        self.X_max = X_max
        self.y_min = y_min
        self.y_max = y_max

        return

    def transform(self, X, y, action):
        """Normalize minibatch

        Parameters
        ----------
        X : 4d/5d tensor
           Minibatch

        direction: int
            1 for rescaling to scaled data, -1 for rescaling to unscaled data

        feature : bool
           If true, X is a feature varible,  else X is a target variable

        Returns:
           X : 4d/5d tensor
           Normalized minibatch

        """
        device = X.device
        
        rang = self.X_max - self.X_min
        bias = self.X_min

        rang = rang.to(device)
        bias = bias.to(device)
        
        if action == 'scale':
            X_tr = (X - bias) / rang
        elif action == 'unscale':
            X_tr = rang * X + bias

        rang = self.y_max - self.y_min
        bias = self.y_min

        device = y.device
        
        rang = rang.to(device)
        bias = bias.to(device)

        if action == 'scale':
            y_tr = (y - bias) / rang
        elif action == 'unscale':
            y_tr = rang * y + bias


        return X_tr, y_tr


    def store(self, args):
        fname = os.path.join(args.h5path, 'minmax.pt')
        tens = {'key': _cache_key(args, self.dataloader.dataset.filenames),
                'vals': torch.tensor([float(self.X_min), float(self.X_max),
                                      float(self.y_min), float(self.y_max)])}
        _atomic_save(tens, fname)

        return

    def load(self, args):
        fname = os.path.join(args.h5path, 'minmax.pt')
        if not os.path.isfile(fname):
            return
        tens = torch.load(fname)
        # a mismatch also rejects pre-fingerprint cache files ('key' absent)
        if tens.get('key') != _cache_key(args,
                                         self.dataloader.dataset.filenames):
            print('Scaler cache is stale (data/filter settings changed); '
                  'refitting.')
            return

        self.X_min = tens['vals'][0].to(self.device)
        self.X_max = tens['vals'][1].to(self.device)
        self.y_min = tens['vals'][2].to(self.device)
        self.y_max = tens['vals'][3].to(self.device)

        return 1


class DummyScaler(object):
    """Datalolader scaler
       

    Parameters
    ----------
    

    Attributes
    ----------
    
    
    """
    
    def __init__(self, dataloader, args):
            
        return


    def fit(self):
        """Determine normalization constants

        Parameters
        ----------
        train_loder : PyTorch DataLoader
           Training DataLoader

        Returns:
           None

        """

        return

    def transform(self, X, y, action):
        """Normalize minibatch

        Parameters
        ----------
        X : 4d/5d tensor
           Minibatch

        direction: int
            1 for rescaling to scaled data, -1 for rescaling to unscaled data

        feature : bool
           If true, X is a feature varible,  else X is a target variable

        Returns:
           X : 4d/5d tensor
           Normalized minibatch

        """

        return X, y

    def store(self, args):
        return

    def load(self, args):
        return


def get_scaler(dataloader, args):

    # === Get Dataset === #
    if args.scaler == 'norm':
        scaler = NormScaler(dataloader, args)
    elif args.scaler == 'minmax':
        scaler = MinmaxScaler(dataloader, args)
    elif args.scaler == 'none':
        scaler = DummyScaler(dataloader, args)

    return scaler

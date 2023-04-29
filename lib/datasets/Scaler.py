
#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################
import torch
from lib.datasets.TurbDataset import TurbDataset

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
        self.X_std = 0.0
        self.y_mean = 0.0
        self.y_std = 0.0

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
            X_mean += torch.sum(X, dim=(0, 2, 3, 4), keepdim=True, dtype=torch.float64)
            y_mean += torch.sum(y, dim=(0, 2, 3, 4), keepdim=True, dtype=torch.float64)
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
            X_std += torch.sum(tmp, dim=(0, 2, 3, 4), keepdim=True, dtype=torch.float64)
            tmp = (y - y_mean) ** 2
            y_std += torch.sum(tmp, dim=(0, 2, 3, 4), keepdim=True, dtype=torch.float64)

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
        
        self.X_max = 0.0
        self.X_min = 0.0
        self.y_max = 0.0
        self.X_min = 0.0

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
        for nbatch, y in enumerate(self.dataloader):
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
            
    
def get_scaler(dataloader, args):

    # === Get Dataset === #
    if args.scaler == 'norm':
        scaler = NormScaler(dataloader, args)
    elif args.scaler == 'minmax':
        scaler = MinmaxScaler(dataloader, args)

    return scaler

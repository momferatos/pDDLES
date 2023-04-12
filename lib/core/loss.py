import torch
import torch.nn as nn
from lib.datasets.TurbDataset import TurbDataset

class Loss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.loss_fn = nn.MSELoss(reduction = 'mean')


    def forward(self, preds, labels):

        loss = self.loss_fn(preds, labels)

        return loss

class SubgridLoss(nn.Module):
    """Loss function that considers only high wavenumbers
       
    Parameters
    ----------
    params : dict
       Dictionary holding global parameters
    

    Attributes
    ----------
    None

    """
    
    def __init__(self, params):
        super(SubgridLoss, self).__init__()

        self.params = params
        self.__name__ = 'SubgridLoss'
        self.loss_fn = nn.MSELoss()
        self.dataset = TurbDataset([], self.params)
        
        return

    def forward(self, y_pred, y):
            
        tens = self.subgrid_scale_tensor(y)
        tens = torch.einsum('bij...,bij...->b...', tens, tens).unsqueeze(1)

        tens_pred = self.subgrid_scale_tensor(y_pred)
        tens_pred = torch.einsum('bij...,bij...->b...',
                            tens_pred,
                            tens_pred).unsqueeze(1)
        
        loss = self.loss_fn(tens_pred, tens)
        
        return loss

    def subgrid_scale_tensor(self, y):

        
        tens1 = torch.einsum('bi...,bj...->bij...', y, y)
        tens2 = self.dataset.LES_filter(tens1)
        
        y_filt = self.dataset.LES_filter(y)
        tens3 = torch.einsum('bi...,bj...->bij...', y_filt, y_filt)
        tens = tens2 - tens3

        ident = torch.eye(3)
        delta = ident.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dims = tens1.shape
        delta = delta.expand(-1, -1, dims[-3], dims[-2], dims[-1])
        trace = torch.einsum('bii...->b...',
                             tens).unsqueeze(1).unsqueeze(2)
        trace = trace.expand(-1, 3, 3, -1, -1, -1)
        delta = delta.to(tens.device)
        tens = tens - 1. / 3. * delta * trace
        
        return tens
    
def get_loss(params):
	
    return SubgridLoss(params)

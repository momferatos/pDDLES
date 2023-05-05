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
    args : Namespace
       Namespace holding global parameters
    

    Attributes
    ----------
    None

    """
    
    def __init__(self, args):
        super(SubgridLoss, self).__init__()

        self.args = args
        self.__name__ = 'SubgridLoss'
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.dataset = TurbDataset([], self.args)
        self.device = args.device
        
        return

    def forward(self, y_pred, y):
            
        tens = self.dataset.subgrid_scale_tensor(y)
        #tens = torch.einsum('bij...,bij...->b...', tens, tens).unsqueeze(1)

        tens_pred = self.dataset.subgrid_scale_tensor(y_pred)
        #tens_pred = torch.einsum('bij...,bij...->b...',
        #                    tens_pred,
        #                    tens_pred).unsqueeze(1)

        # frob = torch.linalg.matrix_norm(tens, dim=(1, 2)).unsqueeze(1)
        # frob_pred = torch.linalg.matrix_norm(tens_pred,
        #                                      dim=(1, 2)).unsqueeze(1)

        #loss = self.loss_fn(tens_pred, tens)
        tens_pred = tens_pred.to(self.device)
        tens = tens.to(self.device)
        
        diffsq_tens = (tens_pred - tens) ** 2
        return diffsq_tens.mean()
    
def get_loss(args):
	
    return SubgridLoss(args)

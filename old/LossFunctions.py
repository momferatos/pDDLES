
#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

import torch
import torch.nn as nn
from lib.datasets.Datasets import TurbDataset

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


        y = y - self.dataset.LES_filter(y)
        y = self.dataset.truncate(y)
        y_pred = y_pred - self.dataset.LES_filter(y_pred)
        y_pred = self.dataset.truncate(y_pred)
        loss = self.loss_fn(y_pred, y)
        
        return loss

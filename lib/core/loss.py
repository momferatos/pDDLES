from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from lib.datasets.TurbDataset import get_helper

class Loss(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self.args = args
        self.loss_fn = nn.MSELoss(reduction = 'mean')


    def forward(self, preds: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:

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

    def __init__(self, args: argparse.Namespace) -> None:
        super(SubgridLoss, self).__init__()

        self.args = args
        self.__name__ = 'SubgridLoss'
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.dataset = get_helper(self.args)
        self.device = args.device

        return

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

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

def get_loss(args: argparse.Namespace) -> SubgridLoss:

    return SubgridLoss(args)

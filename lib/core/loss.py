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

import argparse

import torch
import torch.nn as nn
from lib.datasets.TurbDataset import get_helper, TurbDataset

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
        if self.args.loss == 'sgs_stress':
            tens = self.dataset.subgrid_scale_tensor(y)
            tens_pred = self.dataset.subgrid_scale_tensor(y_pred)
            tens_pred = tens_pred.to(self.device)
            tens = tens.to(self.device)
            diffsq_tens = (tens_pred - tens) ** 2
            return diffsq_tens.mean()
        elif self.args.loss == 'sgs_vel':
            ss_y_pred = y_pred - self.dataset.LES_filter(y_pred)
            ss_y = y - self.dataset.LES_filter(y)
            diffsq_ss = (ss_y_pred - ss_y) ** 2
            return diffsq_ss.mean()
        elif self.args.loss == 'vel':
            return ((y_pred - y) ** 2).mean()
        else:
            raise ValueError(f"Unknown loss function: {self.args.loss}")

def get_loss(args: argparse.Namespace) -> SubgridLoss:

    return SubgridLoss(args)

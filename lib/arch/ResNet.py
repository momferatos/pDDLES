
#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.datasets.TurbDataset import get_helper

class ResNetBlock(nn.Module):
    """A convolutional block of a residual neural network (ResNet)
    
    Parameters
    ----------
    args : Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None

    """
    
    def __init__(self, args: argparse.Namespace) -> None:
        
        super(ResNetBlock, self).__init__()

        self.args = args
        self.conv = args.conv(self.args.num_featmaps,
                                   self.args.num_featmaps,
                                   kernel_size=self.args.kernel_size,
                                   padding='same',
                                   padding_mode='circular', bias=True)
        if self.args.dropout:
            self.dropout = nn.Dropout(self.args.dropout)
        self.batchnorm = self.args.batchnorm(
            num_features=self.args.num_featmaps)
        self.actfun = self.args.actfun

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.conv(x)
        if self.args.dropout:
            out = self.dropout(out)
        out = self.batchnorm(out)
        out = self.actfun(out)
        
        return out + x

class ResNet(nn.Module):
    """Residual neural network (ResNet)

    Parameters
    ----------
    args : Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None

    """
    
    def __init__(self, args: argparse.Namespace) -> None:

        super(ResNet, self).__init__()

        self.args = args
        self.dims = (-3, -2 ,-1)
        self.dataset = get_helper(args)
                                   
        if self.args.scalar:
            self.input_featmaps = 1
        else:
            self.input_featmaps = 2

        self.actfun = self.args.actfun # set activation function
        # first convolutional layer, maps 1 feature map to
        # num_featmaps feature maps
        self.convfirst = self.args.conv(self.input_featmaps,
                                        self.args.num_featmaps,
                                        kernel_size=self.args.kernel_size,
                                        padding='same',
                                        padding_mode='circular',
                                        bias=True)
        self.batchnorm = self.args.batchnorm(
            num_features=self.args.num_featmaps)
        
        # build pipeline of num_blocks ResNetBlocks
        modulelist = nn.ModuleList([])
        for _ in range(self.args.num_blocks):
            modulelist.append(ResNetBlock(self.args))
        self.resnet = nn.Sequential(*modulelist)
        
        # last convolutional layer, maps num_featmaps feature maps
        # back to 1 feature map
        self.convlast = self.args.conv(self.args.num_featmaps,
                                       self.input_featmaps,
                                       kernel_size=self.args.kernel_size,
                                       padding='same',
                                       padding_mode='circular',
                                       bias=True)
        self.batchnormlast = self.args.batchnorm(
            num_features=self.input_featmaps)

        # apply() recurses into every submodule (incl. the convs inside each
        # ResNetBlock); calling init_weights on the modules directly only
        # ever saw the containers, never the conv layers.
        self.apply(self.init_weights)


    def init_weights(self, m: nn.Module) -> None:
        """Kaiming-initializes the convolution weights of the ResNet.

        """
        # data is 3D (Conv3d); the old Conv2d-only check matched nothing.
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # to_helical/from_helical default to the physical domain: they return
        # and expect the real 2-channel helical field and do the FFTs
        # internally, so the CNN pipeline runs directly on their output.
        out = self.dataset.to_helical(x)
        out = self.convfirst(out)
        out = self.batchnorm(out)
        out = self.actfun(out)
        out = self.resnet(out)
        out = self.convlast(out)
        out = self.batchnormlast(out)
        out = self.actfun(out)
        out = self.dataset.from_helical(out)
        out = self.dataset.truncate(out)

        return out

def get_model(args: argparse.Namespace) -> ResNet:

    model = ResNet(args)

    return model

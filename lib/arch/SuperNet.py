
#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

import torch.nn as nn
import torch.nn.functional as F
from Datasets import TurbDataset

class DownNet(nn.Module):
    """Downscaling convolutional layer
    
    Parameters
    ----------
    in_dim: int
       Input length of the side of the data box/cube

    in_dim: int
       Output length of the side of the data box/cube

    args: Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None

    """ 
    def __init__(self, in_dim, out_dim, args):

        super(DownNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.args = args
        self.conv = self.args.conv(self.args.num_channels,
                                        self.args.num_channels,
                                         kernel_size=self.args.kernel_size,
                                         padding='same',
                                         padding_mode='circular',
                                         bias=True)
        self.batchnorm = self.args.batchnorm(
            num_features=self.args.num_channels)
        if self.args.dropout:
            self.dropout = nn.Dropout(self.args.dropout)
        self.actfun = self.args.actfun
        
        return

    
    def forward(self, x):

        mode = 'bicubic' if self.args.dimensions == 2 else 'trilinear'
        out = F.interpolate(x, self.out_dim, mode=mode) # downscale
        out = self.conv(out) # convolve
        if self.args.dropout:
            out = self.dropout(out) # apply dropout
        out = self.batchnorm(out) # apply batch normalization
        out = self.actfun(out) # apply nonlinear activation function
        
        return out

class UpNet(nn.Module):
    """Downscaling convolutional layer
    
    
    Parameters
    ----------
    in_dim: int
       Input length of the side of the data box/cube

    in_dim: int
       Output length of the side of the data box/cube

    args: Namespace
       Namespace holding global parameters


    Attributes
    ----------
    None

    """ 
    def __init__(self, in_dim, out_dim, args):

        super(UpNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.args = args
        self.conv = self.args.conv(self.args.num_channels,
                                        self.args.num_channels,
                                        kernel_size=self.args.kernel_size,
                                        padding='same',
                                        padding_mode='circular',
                                        bias=True)
        if self.args.dropout:
            self.dropout = nn.Dropout(self.args.dropout)
        self.batchnorm = self.args.batchnorm(
            num_features=self.args.num_channels)
        self.actfun = self.args.actfun
        
        return

    
    def forward(self, x):

        mode = 'bicubic' if self.args.dimensions == 2 else 'trilinear'
        out = F.interpolate(x, self.out_dim, mode=mode) # upscale
        out = self.conv(out) # convolve
        if self.args.dropout:
            out = self.dropout(out) # apply dropout
        out = self.batchnorm(out) # apply batch normalization
        out = self.actfun(out) # apply nonlinear activation function
        
        return out

class SuperNet(nn.Module):
    """Super-resolution neural network
    
    Parameters
    ----------
    args: Namespace
       NAmespace holding global parameters

    Attributes
    ----------
    None

    """ 
    def __init__(self, args):

        super(SuperNet, self).__init__()

        self.args = args
        self.conv_first = self.args.conv(
            1, self.args.num_channels,
            kernel_size=self.args.kernel_size,
            padding='same',
            padding_mode='circular',
            bias=True)
        self.conv_last = self.args.conv(
            self.args.num_channels, 1,
            kernel_size=self.args.kernel_size,
            padding='same',
            padding_mode='circular',
            bias=True)
        if self.args.dropout:
            self.dropout = nn.Dropout(self.args.dropout)
        self.batchnorm_first = self.args.batchnorm(
            num_features=self.args.num_channels)
        self.batchnorm_first = self.args.batchnorm(
            num_features=self.args.num_channels)
        self.batchnorm_last = self.args.batchnorm(
            num_features=1)
        self.actfun = self.args.actfun
        
        self.n = self.args.n
        self.num_levels = (self.args.num_levels
                           if self.args.num_levels != 0 else 1)
        self.factor = 2
        ups = []
        downs = []
        for i in range(self.num_levels):
            upnet = UpNet(in_dim=((self.factor ** i) * self.n),
                          out_dim=((self.factor ** (i + 1)) * self.n),
                          args=self.args)
            downnet = DownNet(in_dim=((self.factor ** (i + 1)) * self.n),
                              out_dim=((self.factor ** i) * self.n),
                            args=self.args) 
            ups.append(upnet)
            downs.append(downnet)
            
        downs.reverse()
        nets = ups + downs

        self.supernet = nn.Sequential(*nets)
        self.actfun = self.args.actfun
        self.dataset = TurbDataset([], self.args)
        
        return

    
    def forward(self, x):

        out = self.conv_first(x)
        if self.args.dropout:
            out = self.dropout(out)
        out = self.batchnorm_first(out)
        out = self.actfun(out)
        
        out = self.supernet(out)
        
        out = self.conv_last(out)
        if self.args.dropout:
            out = self.dropout(out)
        out = self.batchnorm_last(out)
        out = self.actfun(out)
        
        out = self.dataset.truncate(out)
        
        return out

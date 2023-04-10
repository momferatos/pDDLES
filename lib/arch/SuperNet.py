
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

    params: dict
       Dictionaty holding global parameters

    Attributes
    ----------
    None

    """ 
    def __init__(self, in_dim, out_dim, params):

        super(DownNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = params
        self.conv = self.params["conv"](self.params["num_channels"],
                                        self.params["num_channels"],
                                         kernel_size=self.params["kernel_size"],
                                         padding='same',
                                         padding_mode='circular',
                                         bias=True)
        self.batchnorm = self.params["batchnorm"](
            num_features=self.params["num_channels"])
        if self.params["dropout"]:
            self.dropout = nn.Dropout(self.params["dropout"])
        self.actfun = self.params["actfun"]
        
        return

    
    def forward(self, x):

        mode = 'bicubic' if self.params["dimensions"] == 2 else 'trilinear'
        out = F.interpolate(x, self.out_dim, mode=mode) # downscale
        out = self.conv(out) # convolve
        if self.params["dropout"]:
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

    params: dict
       Dictionaty holding global parameters


    Attributes
    ----------
    None

    """ 
    def __init__(self, in_dim, out_dim, params):

        super(UpNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = params
        self.conv = self.params["conv"](self.params["num_channels"],
                                        self.params["num_channels"],
                                        kernel_size=self.params["kernel_size"],
                                        padding='same',
                                        padding_mode='circular',
                                        bias=True)
        if self.params["dropout"]:
            self.dropout = nn.Dropout(self.params["dropout"])
        self.batchnorm = self.params["batchnorm"](
            num_features=self.params["num_channels"])
        self.actfun = self.params["actfun"]
        
        return

    
    def forward(self, x):

        mode = 'bicubic' if self.params["dimensions"] == 2 else 'trilinear'
        out = F.interpolate(x, self.out_dim, mode=mode) # upscale
        out = self.conv(out) # convolve
        if self.params["dropout"]:
            out = self.dropout(out) # apply dropout
        out = self.batchnorm(out) # apply batch normalization
        out = self.actfun(out) # apply nonlinear activation function
        
        return out

class SuperNet(nn.Module):
    """Super-resolution neural network
    
    Parameters
    ----------
    params: dict
       Dictionaty holding global parameters

    Attributes
    ----------
    None

    """ 
    def __init__(self, params):

        super(SuperNet, self).__init__()

        self.params = params
        self.conv_first = self.params["conv"](
            1, self.params["num_channels"],
            kernel_size=self.params["kernel_size"],
            padding='same',
            padding_mode='circular',
            bias=True)
        self.conv_last = self.params["conv"](
            self.params["num_channels"], 1,
            kernel_size=self.params["kernel_size"],
            padding='same',
            padding_mode='circular',
            bias=True)
        if self.params["dropout"]:
            self.dropout = nn.Dropout(self.params["dropout"])
        self.batchnorm_first = self.params["batchnorm"](
            num_features=self.params["num_channels"])
        self.batchnorm_first = self.params["batchnorm"](
            num_features=self.params["num_channels"])
        self.batchnorm_last = self.params["batchnorm"](
            num_features=1)
        self.actfun = self.params["actfun"]
        
        self.n = self.params["n"]
        self.num_levels = (self.params["num_levels"]
                           if self.params["num_levels"] != 0 else 1)
        self.factor = 2
        ups = []
        downs = []
        for i in range(self.num_levels):
            upnet = UpNet(in_dim=((self.factor ** i) * self.n),
                          out_dim=((self.factor ** (i + 1)) * self.n),
                          params=self.params)
            downnet = DownNet(in_dim=((self.factor ** (i + 1)) * self.n),
                              out_dim=((self.factor ** i) * self.n),
                            params=self.params) 
            ups.append(upnet)
            downs.append(downnet)
            
        downs.reverse()
        nets = ups + downs

        self.supernet = nn.Sequential(*nets)
        self.actfun = self.params["actfun"]
        self.dataset = TurbDataset([], self.params)
        
        return

    
    def forward(self, x):

        out = self.conv_first(x)
        if self.params["dropout"]:
            out = self.dropout(out)
        out = self.batchnorm_first(out)
        out = self.actfun(out)
        
        out = self.supernet(out)
        
        out = self.conv_last(out)
        if self.params["dropout"]:
            out = self.dropout(out)
        out = self.batchnorm_last(out)
        out = self.actfun(out)
        
        out = self.dataset.truncate(out)
        
        return out

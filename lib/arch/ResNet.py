
#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

import torch.nn as nn
import torch.nn.functional as F
from Datasets import TurbDataset

class ResNetBlock(nn.Module):
    """A convolutional block of a residual neural network (ResNet)
    
    Parameters
    ----------
    params : dict
       Dictionary holding global parameters

    Attributes
    ----------
    None

    """
    
    def __init__(self, params):
        
        super(ResNetBlock, self).__init__()

        self.params = params
        self.conv = params["conv"](self.params["num_channels"],
                                   self.params["num_channels"],
                                   kernel_size=self.params["kernel_size"],
                                   padding='same',
                                   padding_mode='circular', bias=True)
        if self.params["dropout"]:
            self.dropout = nn.Dropout(self.params["dropout"])
        self.batchnorm = self.params["batchnorm"](
            num_features=self.params["num_channels"])
        self.actfun = self.params["actfun"]
        
    def forward(self, x):
        
        out = self.conv(x)
        if self.params["dropout"]:
            out = self.dropout(out)
        out = self.batchnorm(out)
        out = self.actfun(out)
        
        return out + x

class ResNet(nn.Module):
    """Residual neural network (ResNet)

    Parameters
    ----------
    params : dict
       Dictionary holding global parameters

    Attributes
    ----------
    None

    """
    
    def __init__(self, params):

        super(ResNet, self).__init__()

        self.params = params
        self.actfun = self.params["actfun"] # set activation function
        # first convolutional layer, maps 1 channel to num_channels channels
        self.convfirst = self.params["conv"](1, self.params["num_channels"],
                                   kernel_size=self.params["kernel_size"],
                                   padding='same',
                                   padding_mode='circular', bias=True)
        self.batchnorm = self.params["batchnorm"](
            num_features=self.params["num_channels"])
        # build pipeline of num_blocks ResNetBlocks
        self.resnet = nn.Sequential(*(
            self.params["num_blocks"] * [ResNetBlock(self.params)]))
        # last convolutional layer, maps num_channels channels back to 1 channel
        self.convlast = self.params["conv"](self.params["num_channels"], 1,
                                  kernel_size=self.params["kernel_size"],
                                  padding='same',
                                  padding_mode='circular', bias=True)
        self.batchnormlast = self.params["batchnorm"](num_features=1)
        self.dataset = TurbDataset([], self.params)
            
        self.init_weights(self.convfirst)
        self.init_weights(self.resnet)
        self.init_weights(self.convlast)

    
    def init_weights(self, m):
        """Initializes the weights of the ResNet

        """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
            
        return
    
    def forward(self, x):

        out = self.convfirst(x)
        out = self.batchnorm(out)
        out = self.actfun(out)
        out = self.resnet(out)
        out = self.convlast(out)
        out = self.batchnormlast(out)
        out = self.actfun(out)
        out = self.dataset.truncate(out)
        
        return out


#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import ptwt
from lib.datasets.TurbDataset import TurbDataset

class WNet(nn.Module):
    """Wavelet neural network

    Parameters
    ----------
    args : Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None

    """ 
    def __init__(self, args):

        super(WNet, self).__init__()

        self.args = args
        # build pipeline of num_blocks WaveletBlocks
        modulelist = nn.ModuleList([])
        for _ in range(self.args.num_blocks):
            modulelist.append(WaveletBlock(self.args))
        self.waveletnet = nn.Sequential(*modulelist)

#        self.dataset = TurbDataset([], args)
        
        # set number of learnable parameters
        self.device = args.device
        self.dataset = TurbDataset([], args)
        self.dims = (-3, -2, -1)

        self.linear = nn.Linear(1, 1, bias=False) # fully-connected layer
        
        return

    def forward(self, x):

        out = self.dataset.to_helical(x)
        out = self.waveletnet(out)
        out = self.dataset.from_helical(out)
        dims = out.shape
        out = out.view(-1, 1)
        out = self.linear(out)
        out = out.view(dims)
        out = self.dataset.truncate(out)
        
        return out

class WaveletBlock(nn.Module):
    """Wavelet layer

    Parameters
    ----------
    args : Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None
    
    """ 
    def __init__(self, args):
        
        super(WaveletBlock, self).__init__()
        self.args = args
        self.device =self.args.device
        self.dataset = TurbDataset([], args)
        self.dims = (-3, -2, -1)
        
        if args.scalar:
            self.nummods = 1
        else:
            self.nummods = 2

        self.modlist = nn.ModuleList([])
        for nummod in range(self.nummods):
            self.modlist.append(ScalarWaveletBlock(self.args))

        if args.scalar:
            self.batchnorm = torch.nn.BatchNorm3d(num_features=1)
        else:
            self.batchnorm = torch.nn.BatchNorm3d(num_features=3)
            
        self.actfun = args.actfun
        
        return


    def forward(self, x):
                

        tensorlist = []
        for nummod in range(self.nummods):
            tmp = x[:, nummod, :, :, :]
            tmp = self.modlist[nummod](tmp)
            tensorlist.append(tmp.unsqueeze(1))

        out = torch.cat(tensorlist, dim=1)
        out = self.dataset.from_helical(out)
        out = self.batchnorm(out)  
        out = self.actfun(out) 
        out = self.dataset.to_helical(out)
         
        return out + x
                

class ScalarWaveletBlock(nn.Module):
    """Wavelet layer

    Parameters
    ----------
    args : Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None
    
    """ 
    def __init__(self, args):
        
        super(ScalarWaveletBlock, self).__init__()

        self.args = args
        self.device = args.device
        self.wavelet_type = args.wavelet
        self.actfun = args.actfun # define activation function
        self.wavelet = pywt.Wavelet(self.wavelet_type) # define wavelet
        self.n = self.args.n # define resolution of the data square/cube
        # define number of levels of the wavelet transform
        if args.num_levels:
            self.num_levels = args.num_levels
        else:
            self.num_levels = pywt.dwt_max_level(args.n, self.wavelet)
            
        # define wavelet coeffient multiplication mode
        self.mode = args.wavelet_mode 
        self.dummy_param = nn.Parameter(torch.empty(0), requires_grad=True)
        self.dimensions = args.dimensions
                
        # create a sample input
        if args.dimensions == 2:        
            X = torch.rand((1, self.n, self.n),
                           dtype=torch.float32)
        else:
            X = torch.rand((1, self.n, self.n, self.n),
                           dtype=torch.float32)

        # forward discrete wavelet transform
        if args.dimensions == 2:
            wavecoeffs = ptwt.wavedec2(X, self.wavelet, mode='periodic',
                                       level=self.num_levels)
        else:
            wavecoeffs = ptwt.wavedec3(X, self.wavelet, mode='periodic',
                                       level=self.num_levels)
                       
        # create structure of learnable multiplication coefficients
        params_list = nn.ParameterList([])
        for wavecoeff in wavecoeffs:
            if type(wavecoeff) == type(dict()):
                params_dict = nn.ParameterDict({})
                for wavecoeff_key in wavecoeff.keys():
                    if self.mode == 'all':
                        tmp = torch.rand_like(wavecoeff[wavecoeff_key],
                                              dtype=torch.float32)
                    elif self.mode == 'one':
                        tmp = torch.rand(1, dtype=torch.float32)
                    elif self.mode == 'outer':
                        tmp = torch.rand(wavecoeff[wavecoeff_key].shape[-1],
                                         dtype=torch.float32)
                    params_dict[wavecoeff_key] = nn.Parameter(
                        tmp, requires_grad=True)
                    nn.init.uniform_(params_dict[wavecoeff_key])
                params_list.append(params_dict)
            else:
                params_tuple = nn.ParameterList([])
                for wv in wavecoeff:
                    if self.mode == 'all':
                        tmp = torch.rand_like(wv,
                                              dtype=torch.float32)
                    elif self.mode == 'one':
                        tmp = torch.rand(1, dtype=torch.float32)
                    elif self.mode == 'outer':
                        tmp = torch.rand(wv.shape[-1], dtype=torch.float32)
                    params_tmp = nn.Parameter(tmp, requires_grad=True)
                    nn.init.uniform_(params_tmp)
                    params_tuple.append(params_tmp)
                params_tuple = nn.ParameterList(params_tuple)
                params_list.append(params_tuple)

        self.params_list = params_list
        
        return
    
    def forward(self, x):

        # forward discrete wavelet transform
        if self.dimensions == 2:
            wavecoeffs = ptwt.wavedec2(x, self.wavelet, mode='periodic',
                                       level=self.num_levels)
        else:
            wavecoeffs = ptwt.wavedec3(x, self.wavelet, mode='periodic',
                                       level=self.num_levels)
            
        # multiply wavelet coefficients by learnable parameters
        wavecoeffs_out = []
        for i,(wavecoeff,param) in enumerate(zip(wavecoeffs,
                                                 self.params_list)):
            if i == 0:
                wavecoeff_out = wavecoeff
            elif type(wavecoeff) != type(dict()):
                wavecoeff_out = []
                for wv, pm in zip(wavecoeff, param):
                    if self.mode == 'outer':
                        mult = torch.outer(pm, pm)
                    else:
                        mult = pm
                    wavecoeff_out.append(wv * mult)
            else:
                wavecoeff_out = dict()
                for wv_key, wv_val in wavecoeff.items():
                    if self.mode == 'outer':
                        pm = param[wv_key]
                        mult = torch.outer(pm, pm)
                    else:
                        mult = param[wv_key]
                    wavecoeff_out[wv_key] = wv_val * mult
            wavecoeffs_out.append(wavecoeff_out)


        # inverse discrete wavelet transform
        if self.dimensions == 2:
            out = ptwt.waverec2(wavecoeffs_out, self.wavelet)
        else:
            out = ptwt.waverec3(wavecoeffs_out, self.wavelet)
                    
        return out

def get_model(args):

    model = WNet(args)

    return model

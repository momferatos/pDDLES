
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

class WaveletNet(nn.Module):
    """Wavelet neural network

    Parameters
    ----------
    params : dict
       Dictionary holding global parameters

    Attributes
    ----------
    None

    """ 
    def __init__(self, params):

        super(WaveletNet, self).__init__()

        self.num_levels = (params["num_levels"] if params["num_levels"] != 0
                           else int(np.log2(n)))
        # define WaveletBlock
        self.block = WaveletBlock(params)

        # build pipeline of num_blocks WaveletBlocks
        self.waveletnet = nn.Sequential(
            *(params["num_blocks"] * [self.block]))

#        self.dataset = TurbDataset([], params)
        
        # set number of learnable parameters
        self.num_params = self.block.num_params
        
        return

    def forward(self, x):

        # squeeze 1st dimension because of ptwt requirement
        out = x.squeeze(1)
        out = self.waveletnet(out)
        # unsqueeze back 1st dimension
        out  = out.unsqueeze(1)
        # truncate everything beyond DNS resolution 
#        out = self.dataset.truncate(out)
        
        return out

class WaveletBlock(nn.Module):
    """Wavelet layer

    Parameters
    ----------
    params : dict
       Dictionary holding global parameters

    Attributes
    ----------
    None
    
    """ 
    def __init__(self, params):
        
        super(WaveletBlock, self).__init__()

        self.wavelet_type = params["wavelet_type"]
        self.actfun = params["actfun"] # define activation function
        self.wavelet = pywt.Wavelet(self.wavelet_type) # define wavelet
        self.n = params['n'] # define resolution of the data square/cube
        # define number of levels of the wavelet transform
        self.num_levels = params["num_levels"]
        # define wavelet coeffient multiplication mode
        self.mode = params["wavelet_mode"] 
        self.dummy_param = nn.Parameter(torch.empty(0), requires_grad=True)
        self.dimensions = params["dimensions"]
        
        # create a sample input
        if params["dimensions"] == 2:        
            X = torch.rand((1, self.n, self.n),
                           dtype=torch.float32)
        else:
            X = torch.rand((1, self.n, self.n, self.n),
                           dtype=torch.float32)

        # forward discrete wavelet transform
        if params["dimensions"] == 2:
            wavecoeffs = ptwt.wavedec2(X, self.wavelet, mode='periodic',
                                       level=self.num_levels)
        else:
            wavecoeffs = ptwt.wavedec3(X, self.wavelet, mode='periodic',
                                       level=self.num_levels)
                       
        # create structure of learnable multiplication coefficients
        self.num_params = 0
        params_list = []
        for wavecoeff in wavecoeffs:
            if type(wavecoeff) == type(dict()):
                params_tmp = dict()
                for wavecoeff_key in wavecoeff.keys():
                    if self.mode == 'all':
                        tmp = torch.rand_like(wavecoeff[wavecoeff_key],
                                              dtype=torch.float32)
                    elif self.mode == 'one':
                        tmp = torch.rand(1, dtype=torch.float32)
                    elif self.mode == 'outer':
                        tmp = torch.rand(wavecoeff[wavecoeff_key].shape[-1],
                                         dtype=torch.float32)
                    self.num_params += tmp.numel()
                    params_tmp[wavecoeff_key] = nn.Parameter(
                        tmp, requires_grad=True)
                    nn.init.uniform_(params_tmp[wavecoeff_key])
                params_dict = nn.ParameterDict(params_tmp)
                params_list.append(params_dict)
            else:
                params_tuple = []
                for wv in wavecoeff:
                    if self.mode == 'all':
                        tmp = torch.rand_like(wv,
                                              dtype=torch.float32)
                    elif self.mode == 'one':
                        tmp = torch.rand(1, dtype=torch.float32)
                    elif self.mode == 'outer':
                        tmp = torch.rand(wv.shape[-1], dtype=torch.float32)
                    self.num_params += tmp.numel()
                    params_tmp = nn.Parameter(tmp, requires_grad=True)
                    nn.init.uniform_(params_tmp)
                    params_tuple.append(params_tmp)
                params_tuple = nn.ParameterList(tuple(params_tuple))
                params_list.append(params_tuple)

        self.params_list = nn.ParameterList(params_list)
        
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
            
        out = self.actfun(out) # apply activation function

        out = out.squeeze(1)
        
        
        return out

def get_model(params):

    model = WaveletNet(params)

    return model

#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets.TurbDataset import TurbDataset

class FourierNet(nn.Module):
    """Fourier neural network

    Parameters
    ----------
    params : dict
       Dictionary holding global parameters

    Attributes
    ----------
    None

    """ 
    def __init__(self, params):

        super(FourierNet, self).__init__()
        self.params = params
        # self.dataset = TurbDataset([], self.params)
        
        # build pipeline of num_blocks FourierBlocks
        self.fouriernet = nn.Sequential(
            *(self.params["num_blocks"] * [FourierBlock(self.params)])) 
        
        return

    def forward(self, x):
        
        out = self.fouriernet(x)
        # out = self.dataset.truncate(out)
        
        return out

class FourierBlock(nn.Module):
    """Fourier layer

    Parameters
    ----------
    params : dict
       Dictionary holding global parameters

    Attributes
    ----------
    None
    
    """ 
    def __init__(self, params):

        super(FourierBlock, self).__init__()

        self.params = params
        self.actfun = self.params["actfun"] # select activation function

        # Create learnable spectral multiplication coefficients
        alpha = torch.rand(params["num_coeffs"],
                           dtype=torch.float32)
        self.alpha = nn.Parameter(alpha, requires_grad=True)
        nn.init.uniform_(self.alpha)

        self.linear = nn.Linear(1, 1) # fully-connected layer
        
        wvs = torch.fft.fftfreq(self.params["n"]) # wavenumbers
        # wavenumbers in the real-to-half-complex dimension (dim=-1)
        rwvs = torch.fft.rfftfreq(self.params["n"])
        # wavevector magnitudes
        # 2d
        wvs2d = torch.sqrt(wvs.reshape(-1, 1) ** 2 +
                           rwvs.reshape(1, -1) ** 2)
        # 3d
        wvs3d = torch.sqrt(wvs.reshape(-1, 1, 1) ** 2 +
                           wvs.reshape(1, -1, 1) ** 2 +
                           rwvs.reshape(1, 1, -1) ** 2)
        wvms = wvs2d if params["dimensions"] == 2 else wvs3d
        # maximum wavenumber
        wvmax = torch.max(wvms)
        # define the indices in which the learnable spectral coefficients
        # will be multiplied
        wvs_spec = torch.linspace(0.0, wvmax, self.params["num_coeffs"])
        # create 2d/3d view
        wvs_spec = (wvs_spec.view(-1, 1, 1) 
                    if params["dimensions"] == 2
                    else wvs_spec.view(-1, 1, 1, 1))
        diffs = torch.abs(wvms - wvs_spec)
        self.idxs = torch.argmin(diffs, dim=0)
        self.idxs = self.idxs.unsqueeze(0).unsqueeze(0)
        
        return

    
    def forward(self, x):

        dims = x.shape
        
        # forward FFT
        if self.params["dimensions"] == 2:
            out = torch.fft.rfftn(x, dim=(2, 3), norm='ortho')
        else:
            out = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')


        # multiplication by learnable spectral coeficients
        out = self.alpha[self.idxs] * out
        
        # inverse FFT
        if self.params["dimensions"] == 2:
            out = torch.fft.irfftn(out, dim=(2, 3), norm='ortho')
        else:
            out = torch.fft.irfftn(out, dim=(2, 3, 4), norm='ortho')
            
        out = self.actfun(out)
        out = out.view(-1, 1)
        out = self.linear(out)
        out = out.view(dims)
                
        return out

def get_model(params):

    model = FourierNet(params)

    return model

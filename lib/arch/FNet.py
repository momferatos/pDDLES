#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets.TurbDataset import TurbDataset

class FNet(nn.Module):
    """Fourier neural network

    Parameters
    ----------
    args : Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None

    """ 
    def __init__(self, args):

        super(FNet, self).__init__()
        self.args = args
        self.dataset = TurbDataset([], self.args)
        
        # build pipeline of num_blocks FourierBlocks
        modulelist = nn.ModuleList([])
        for _ in range(self.args.num_blocks):
            modulelist.append(FourierBlock(self.args))
        self.fouriernet = nn.Sequential(*modulelist)

        self.linear = nn.Linear(1, 1, bias=False) # fully-connected layer
        
        return

    def forward(self, x):
        
        out = self.dataset.to_helical(x, outdomain='fourier')
        out = self.fouriernet(out)
        out = self.dataset.from_helical(out, indomain='fourier')
        dims = out.shape
        out = out.view(-1, 1)
        out = self.linear(out)
        out = out.view(dims)
        out = self.dataset.truncate(out)
        
        return out

class FourierBlock(nn.Module):
    """Fourier layer

    Parameters
    ----------
    args : Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None
    
    """ 
    def __init__(self, args):

        super(FourierBlock, self).__init__()

        self.dims = (-3, -2, -1)
        self.args = args
        self.actfun = self.args.actfun # select activation function
        self.dataset = TurbDataset([], self.args)
        
        # Create learnable spectral multiplication coefficients
        self.alpha = torch.rand(args.num_coeffs,
                                dtype=torch.float32)
        self.alpha = nn.Parameter(self.alpha, requires_grad=True)
        nn.init.uniform_(self.alpha)

        if args.scalar:
            self.batchnorm = self.args.batchnorm(
                num_features=1)
        else:
            self.batchnorm = self.args.batchnorm(
                num_features=3)
        
        wvs = torch.fft.fftfreq(self.args.n) # wavenumbers
        # wavenumbers in the real-to-half-complex dimension (dim=-1)
        rwvs = torch.fft.rfftfreq(self.args.n)
        # wavevector magnitudes
        # 2d
        wvs2d = torch.sqrt(wvs.reshape(-1, 1) ** 2 +
                           rwvs.reshape(1, -1) ** 2)
        # 3d
        wvs3d = torch.sqrt(wvs.reshape(-1, 1, 1) ** 2 +
                           wvs.reshape(1, -1, 1) ** 2 +
                           rwvs.reshape(1, 1, -1) ** 2)
        wvms = wvs2d if args.dimensions == 2 else wvs3d
        # maximum wavenumber
        wvmax = torch.max(wvms)
        # define the indices in which the learnable spectral coefficients
        # will be multiplied
        wvs_spec = torch.linspace(0.0, wvmax, self.args.num_coeffs)
        # create 2d/3d view
        wvs_spec = (wvs_spec.view(-1, 1, 1) 
                    if args.dimensions == 2
                    else wvs_spec.view(-1, 1, 1, 1))
        diffs = torch.abs(wvms - wvs_spec)
        self.idxs = torch.argmin(diffs, dim=0)
        self.idxs = self.idxs.unsqueeze(0).unsqueeze(0)
        
        return

    
    def forward(self, x):
        
        
        out = self.alpha[self.idxs] * x
        out = self.dataset.from_helical(out, indomain='fourier')
        out = self.batchnorm(out)
        out = self.actfun(out)
        out = self.dataset.to_helical(out, outdomain='fourier')

        if not self.args.noskip:
            out = out + x
            
        return out

def get_model(args):

    model = FNet(args)

    return model

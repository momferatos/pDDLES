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
    args : Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None

    """ 
    def __init__(self, args):

        super(FourierNet, self).__init__()
        self.args = args
        self.dataset = TurbDataset([], self.args)
        self.linear = nn.Linear(1, 1, bias=False) # fully-connected layer
        # build pipeline of num_blocks FourierBlocks
        modulelist = nn.ModuleList([])
        for nblock in range(self.args.num_blocks):
            modulelist.append(FourierBlock(self.args))
            
        self.fouriernet = nn.Sequential(*modulelist)
        
        return

    def forward(self, x):
        
        dims = x.shape
        out = self.fouriernet(x)
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

        self.args = args
        self.actfun = self.args.actfun # select activation function

        # Create learnable spectral multiplication coefficients
        self.alpha = torch.rand(args.num_coeffs,
                           dtype=torch.float32)
        self.alpha = nn.Parameter(self.alpha, requires_grad=True)
        nn.init.uniform_(self.alpha)

        # self.linear = nn.Linear(1, 1) # fully-connected layer

        self.batchnorm = self.args.batchnorm(
        num_features=4)
        
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

        dims = x.shape
        
        # forward FFT
        if self.args.dimensions == 2:
            out = torch.fft.rfftn(x, dim=(2, 3), norm='ortho')
        else:
            out = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')


        # multiplication by trainable spectral coeficients
        out = self.alpha[self.idxs] * out
        
        # inverse FFT
        if self.args.dimensions == 2:
            out = torch.fft.irfftn(out, dim=(2, 3), norm='ortho')
        else:
            out = torch.fft.irfftn(out, dim=(2, 3, 4), norm='ortho')

        # batch normalization
        out = self.batchnorm(out)

        out = self.actfun(out)
        #out = out.view(-1, 1)
        #out = self.linear(out)
        #out = out.view(dims)
                
        return out + x

def get_model(args):

    model = FourierNet(args)

    return model

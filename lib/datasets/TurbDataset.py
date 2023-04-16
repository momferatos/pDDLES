
#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import sys

class TurbDataset(Dataset):
    """Dataset representing a shapshot from the time evolution of the passive
       scalar field

    Parameters
    ----------
    filenames : list of strs
       List containing the HDF5 input files

    args: Namespace
       Namespace holding global parameters

    Attributes
    ----------
    None
    
    """
    
    def __init__(self, filenames, args):
        self.filenames = filenames # HDF5 filenames' list
        self.args = args
        self.eps = 1.0e-5
        self.device = args.device
        
        if not filenames:
            return
        
        filename = self.filenames[0]
        # load data from HDF5 file
        with h5py.File(filename, 'r') as h5file:
            # 2d or 3d data, using float32 for better performance on the GPU
            X = np.array(h5file[self.args.hdf5_key], dtype='float32')

        if self.args.hdf5_key == 'u':
            X = X.transpose([3, 2, 1, 0])
            
        X = torch.from_numpy(X)
        X = X.unsqueeze(0)
        
        self.dims = (-3, -2, -1)
        
        k, kappa = self.wave_vectors(X)

        k = k.to(self.device)
        kappa = kappa.to(self.device)
        
        ex = torch.Tensor(
            [1.0, 0.0, 0.0]).unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1)
        ex = ex.expand(-1, kappa.shape[1],
                           kappa.shape[2], kappa.shape[3])
        ex = ex.to(self.device)

        ey = torch.Tensor(
            [0.0, 1.0, 0.0]).unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1)
        ey = ey.expand(-1, kappa.shape[1],
                           kappa.shape[2], kappa.shape[3])
        ey = ey.to(self.device)
        
        ez = torch.Tensor(
            [0.0, 0.0, 1.0]).unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1)
        ez = ez.expand(-1, kappa.shape[1],
                           kappa.shape[2], kappa.shape[3])
        ez = ez.to(self.device)

        ezxk = torch.cross(ez, k, dim=0)
        kxezxk = torch.cross(k, ezxk, dim=0)
        mag = torch.linalg.vecdot(ezxk, ezxk, dim=0)
        #mag = torch.where(torch.abs(mag) < self.eps, 1.0, mag)
        mag_ezxk = torch.sqrt(mag)
        mag = torch.linalg.vecdot(kxezxk, kxezxk, dim=0)
        #mag = torch.where(torch.abs(mag) < self.eps, 1.0, mag)
        mag_kxezxk = torch.sqrt(mag)
        mask = (mag_ezxk != 0.0)
        sqrt2 = np.sqrt(2.0)
        
        self.hplus = torch.where(mask, ezxk / (sqrt2 * mag_ezxk) + 1j * kxezxk / (sqrt2 * mag_kxezxk), (ex + 1j * ey) / sqrt2)

        self.hminus = torch.where(mask, ezxk / (sqrt2 * mag_ezxk) - 1j * kxezxk / (sqrt2 * mag_kxezxk), (ex - 1j * ey) / sqrt2)
        
        return

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx] 

        # load data from HDF5 file
        with h5py.File(filename, 'r') as h5file:
            # 2d or 3d data, using float32 for better performance on the GPU
            y = np.array(h5file[self.args.hdf5_key], dtype='float32')
            # number of file 
            num_file = 0 #int(np.array(h5file['nfile'])[0])
            # time instant
            time = float(np.array(h5file['time'])[0])

        if self.args.hdf5_key == 'u':
            y = y.transpose([3, 2, 1, 0])
            
        y = torch.from_numpy(y)
        # truncate to Patterson-Orszag dealiasing limit
        y = self.truncate(y) 
        if self.args.hdf5_key == 'scl':
            y = y.unsqueeze(0) # Add extra tensor dimension required by PyTorch
                
        return y


    def truncate(self, x):
        """Truncate a batch to DNS resolution

        Parameters
        ----------
        x : 4d or 5d tensor
           Minibatch

        Returns
        ----------
        x : 4d or 5d tensor
           Truncated minibatch

        """

        n = x.shape[-2] # get DNS square linear resolution
        dims = (-2, -1) if self.args.dimensions == 2 else (-3, -2, -1)
        wvs = torch.fft.fftfreq(n) # wavenumbers
        rwvs = torch.fft.rfftfreq(n) # wavenumbers of real-to-half-complex dim
        # define wavevector magnintudes
        wvs2d = torch.sqrt(wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvs3d = torch.sqrt(wvs.view(-1, 1, 1) ** 2 +
                           wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvms = wvs2d if self.args.dimensions == 2 else wvs3d
        kmax = wvms.max() # maximum wavevector
        beta = np.sqrt(2.0) / 3.0 # truncation factor
        # apply truncation mask
        mask = wvms > beta * kmax
        mask = mask.squeeze(0)
        fout = torch.fft.rfftn(x, dim=dims, norm='ortho')
        fout[..., mask] = 0.0
        out = torch.fft.irfftn(fout, dim=dims, norm='ortho')


        return out


    def LES_filter(self, y):
        """Apply a sharp cutoff spectral LES filter

        Parameters
        ----------
        y : 4d/5d tensor
           Minibatch

        Returns
        ----------
        y : 4d/5d tensor
           Filtered minibatch

        """

        n = y.shape[-2] # get DNS square linear resolution
        # dims = (-2, -1) if self.args.dimensions == 2 else (-3, -2, -1)
        dims = (-3, -2, -1)
        fy = torch.fft.rfftn(y, dim=dims, norm='ortho') # forward real-to-half-complex FFT
        wvs = torch.fft.fftfreq(n) # wavenumbers
        rwvs = torch.fft.rfftfreq(n) # wavenumbers of real-to-half-complex dim
        # wavevector magnitudes
        wvs2d = torch.sqrt(wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvs3d = torch.sqrt(wvs.view(-1, 1, 1) ** 2 +
                           wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvms = wvs2d if self.args.dimensions == 2 else wvs3d
        wvmax = torch.max(wvms) # maximum wavevector magnitude
        mask = wvms > self.args.alpha * wvmax # define filter mask
        fy[..., mask] = 0.0 # apply filter
        y = torch.fft.irfftn(fy, dim=dims, norm='ortho') # inverse half-complex-to-real FFT

        return y

    def wave_vectors(self, X):

        n = X.shape[-2]
        dims = (-3, -2, -1)
        wvs = torch.fft.fftfreq(n) # wavenumbers
        rwvs = torch.fft.rfftfreq(n) # wavenumbers of real-to-half-complex dim

        # wavevectors
        k1, k2, k3 = torch.meshgrid([wvs, wvs, rwvs], indexing='ij')
        kmag = torch.sqrt(k1**2 + k2**2 + k3**2)
        kmag = torch.where(torch.abs(kmag) < self.eps, 1.0, kmag)
        kappa1 = k1 / kmag
        kappa2 = k2 / kmag
        kappa3 = k3 / kmag
        
        # batch of wavevectors
        k = torch.stack([k1, k2, k3])

        # batch of unit vectors
        kappa = torch.stack([kappa1, kappa2, kappa3])

        k = k.to(self.device)
        kappa = kappa.to(self.device)
        return k, kappa

    def helical_checks(self, X):

        k, kappa = self.wave_vectors(X)
        
        d00 = torch.einsum('i...,i...',
                           torch.conj_physical(self.hplus),
                           self.hplus)
        
        d00 = d00.abs().max()
        
        d01 = torch.einsum('i...,i...',
                           torch.conj_physical(self.hplus),
                           self.hminus)
        d01 = d01
        d01 = d01.abs().max()
        
        d10 = torch.einsum('i...,i...',
                           torch.conj_physical(self.hminus),
                           self.hplus)
        d10 = d10
        d10 = d10.abs().max()
        
        d11 = torch.einsum('i...,i...',
                           torch.conj_physical(self.hminus),
                           self.hminus)
        mask = (d11 != 0.0)
        d11 = d11.abs().max()
        
        print('delta: ', torch.Tensor([[d00, d01], [d10, d11]]))

        
        return
    
    def to_helical(self, X):
    
        fX = torch.fft.rfftn(X, dim=self.dims, norm='ortho') 
        
        hplus = self.hplus.unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        hplus = hplus.to(self.device)
        hminus = self.hminus.unsqueeze(0).expand(
            X.shape[0], -1, -1, -1, -1)
        hminus = hminus.to(self.device)
        
        # mag = torch.einsum('bi...,bi...->b...',
        #                    (hplus),
        #                    hplus).unsqueeze(1)
        # mag = torch.where(torch.abs(mag) < self.eps, 1.0, torch.sqrt(mag))
        # mag = 1.0
        # hplus = hplus / mag

        # mag = torch.einsum('bi...,bi...->b...',
        #                    (hminus),
        #                    hminus).unsqueeze(1)
        # mag = torch.where(torch.abs(mag) < self.eps, 1.0, torch.sqrt(mag))
        # mag = 1.0
        # hminus = hminus / mag
        
        aplus = torch.linalg.vecdot(hplus, fX, dim=1).unsqueeze(1)
        aminus = torch.linalg.vecdot(hminus, fX, dim=1).unsqueeze(1)

        apm = torch.cat((aplus.real, aplus.imag,
                         aminus.real, aminus.imag), dim=1)

        apm = torch.fft.irfftn(apm, dim=self.dims, norm='ortho')
        
        return apm


    def from_helical(self, apm):

        apm = torch.fft.rfftn(apm, dim=self.dims, norm='ortho')
        
        aplus = apm[:, 0, :, :, :].unsqueeze(1) + 1j * apm[:, 1, :, :, :].unsqueeze(1)
        aminus = apm[:, 2, :, :, :].unsqueeze(1) + 1j * apm[:, 3, :, :, :].unsqueeze(1)

        hplus = self.hplus.unsqueeze(0).expand(apm.shape[0], -1, -1, -1, -1)
        hplus = hplus.to(self.device)
        hminus = self.hminus.unsqueeze(0).expand(
            apm.shape[0], -1, -1, -1, -1)
        hminus = hminus.to(self.device)
        
        apm = aplus * hplus + aminus * hminus

        X = torch.fft.irfftn(apm, dim=self.dims, norm='ortho')
        
        return X

    def divergence(self, X):

        k, _ = self.wave_vectors(X)

        k = k.unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        dims = (-3, -2, -1)
        fX = torch.fft.rfftn(X, dim=dims, norm='ortho') 
        div = torch.linalg.vecdot(1j * k, fX, dim=1)
        div = torch.fft.irfftn(div, dim=dims, norm='ortho')
        div = div.abs().max()
        
        return div

    def vorticity(self, X):
        
        k, _ = self.wave_vectors(X)
        
        k = k.unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        dims = (-3, -2, -1)
        fX = torch.fft.rfftn(X, dim=dims, norm='ortho') # forward real-to-half-complex FF
        w = torch.cross(1j * k, fX, dim=1)
        w = torch.fft.irfftn(w, dim=dims, norm='ortho')
        
        return w

    def subgrid_scale_tensor(self, y):

        
        tens1 = torch.einsum('bi...,bj...->bij...', y, y)
        tens2 = self.LES_filter(tens1)
        
        y_filt = self.LES_filter(y)
        tens3 = torch.einsum('bi...,bj...->bij...', y_filt, y_filt)
        tens = tens2 - tens3

        ident = torch.eye(3)
        delta = ident.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dims = tens1.shape
        delta = delta.expand(-1, -1, dims[-3], dims[-2], dims[-1])
        trace = torch.einsum('bii...->b...',
                             tens).unsqueeze(1).unsqueeze(2)
        trace = trace.expand(-1, 3, 3, -1, -1, -1)
        delta = delta.to(self.device)
        tens = tens - 1. / 3. * delta * trace

                    
        return tens
    
    
def get_dataset(filenames, args):

    # === Get Dataset === #
    train_dataset = TurbDataset(filenames, args)

    return train_dataset

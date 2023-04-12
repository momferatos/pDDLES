
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

    params: dict
       Dictionary holding global parameters

    Attributes
    ----------
    None
    
    """
    
    def __init__(self, filenames, params):
        self.filenames = filenames # HDF5 filenames' list
        self.params = params
        self.eps = 1.0e-5

        if not filenames:
            return
        
        filename = self.filenames[0]
        # load data from HDF5 file
        with h5py.File(filename, 'r') as h5file:
            # 2d or 3d data, using float32 for better performance on the GPU
            X = np.array(h5file[self.params["hdf5_key"]], dtype='float32')

        if self.params["hdf5_key"] == 'u':
            X = X.transpose([3, 2, 1, 0])
            
        X = torch.from_numpy(X)
        X = X.unsqueeze(0)
        device = X.device
        self.dims = (-3, -2, -1)
        
        k, kappa = self.wave_vectors(X)

        ex = torch.Tensor(
            [1.0, 0.0, 0.0]).unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1)
        ex = ex.expand(-1, kappa.shape[1],
                           kappa.shape[2], kappa.shape[3])
        ex = ex.to(device)

        ey = torch.Tensor(
            [0.0, 1.0, 0.0]).unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1)
        ey = ey.expand(-1, kappa.shape[1],
                           kappa.shape[2], kappa.shape[3])
        ey = ey.to(device)
        
        ez = torch.Tensor(
            [0.0, 0.0, 1.0]).unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1)
        ez = ez.expand(-1, kappa.shape[1],
                           kappa.shape[2], kappa.shape[3])
        ez = ez.to(device)

        ezxk = torch.cross(ez, k, dim=0)
        kxezxk = torch.cross(k, ezxk, dim=0)
        mag = torch.einsum('i...,i...', ezxk, ezxk)
        #mag = torch.where(torch.abs(mag) < self.eps, 1.0, mag)
        mag_ezxk = torch.sqrt(mag)
        mag = torch.einsum('i...,i...', kxezxk, kxezxk)
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
            y = np.array(h5file[self.params["hdf5_key"]], dtype='float32')
            # number of file 
            num_file = 0 #int(np.array(h5file['nfile'])[0])
            # time instant
            time = float(np.array(h5file['time'])[0])

        if self.params["hdf5_key"] == 'u':
            y = y.transpose([3, 2, 1, 0])
            
        y = torch.from_numpy(y)
        # truncate to Patterson-Orszag dealiasing limit
        y = self.truncate(y) 
        if self.params["hdf5_key"] == 'scl':
            y = y.unsqueeze(0) # Add extra tensor dimension required by PyTorch
                
        return y

    def determine_norm_constants(self, train_loader):
        """Determine normalization constants

        Parameters
        ----------
        train_loder : PyTorch DataLoader
           Training DataLoader

        Returns:
           None

        """

        
                
        if self.params["norm_mode"] == 'mean_std':

            
            # self.X_mean = 0.0
            # self.X_std = 1.0
            # self.y_mean = 0.0
            # self.y_std = 1.0
            # return

            X_mean = 0.0
            y_mean = 0.0
            fac = 0.0
            nbatches = len(train_loader)
            for nbatch, y in enumerate(train_loader):
                print(f'Computing mean: {nbatch}/{nbatches}')
                # self.helical_checks(y[-1])
                # y1 = self.to_helical(y)
                # y2 = self.from_helical(y1)
                # diff = y - y2
                # print('diff:', diff.abs().max(), diff.abs().mean())
                # print('input: ', self.divergence(y).abs().max())
                # print('output: ', self.divergence(y2).abs().max())
                #sys.exit(0)
                
                X = self.LES_filter(y)

                if self.params["prediction_mode"] == 'large_to_small':
                    y = y - X

                tmp = torch.einsum('bijkl->i', X)
                X_mean += torch.sqrt(torch.dot(tmp, tmp)).item()
                tmp = torch.einsum('bijkl->i', y)
                X_mean += torch.sqrt(torch.dot(tmp, tmp)).item()
                fac += X.numel()

            X_mean /= fac
            y_mean /= fac

            X_std = 0.0
            y_std = 0.0
            fac = 0.0
            for nbatch, y in enumerate(train_loader):
                print(f'Computing std: {nbatch}/{nbatches}')
#                y = y.to(g.device)
                X = self.LES_filter(y)

                if self.params["prediction_mode"] == 'large_to_small':
                    y = y - X

                tmp = (X - X_mean) ** 2
                X_std += tmp.sum()
                tmp = (y - y_mean) ** 2
                y_std += tmp.sum()
                
                fac += X.numel()

            X_std /= fac
            y_std /= fac

            X_std = torch.sqrt(X_std)
            y_std = torch.sqrt(y_std)

            self.X_mean = X_mean
            self.X_std = X_std
            self.y_mean = y_mean
            self.y_std = y_std
            

        elif self.params["norm_mode"] == 'min_max':
            # calculate maximum/minimum of train dataset for normalization 
            y_min = 1.0e6
            y_max = -1.0e6
            X_min = 1.0e6
            X_max = -1.0e6
            for nbatch, yy in enumerate(train_loader):
                y,_,_ = yy
#                y = y.to(g.device)
                X = LES_filter(y, self.params["alpha"])

                if self.params["prediction_mode"] == 'large_to_small':
                    y = y - X

                X_min = min(X_min, X.flatten().min())
                X_max = max(X_max, X.flatten().max())
                y_min = min(y_min, y.flatten().min())
                y_max = max(y_max, y.flatten().max())

                self.X_min = X_min
                self.X_max = X_max
                self.y_min = y_min
                self.y_max = y_max

        else:

            print(f'Invalid self.params["norm_mode"] '
                  f'{self.params["norm_mode"]}')
            exit(1)

        return


    def normalize(self, X, direction, feature=True):
        """Normalize minibatch

        Parameters
        ----------
        X : 4d/5d tensor
           Minibatch

        direction: int
            1 for rescaling to scaled data, -1 for rescaling to unscaled data

        feature : bool
           If true, X is a feature varible,  else X is a target variable

        Returns:
           X : 4d/5d tensor
           Normalized minibatch

        """

        if feature:
            if self.params["norm_mode"] == 'mean_std':
                rang = self.X_std
                bias = self.X_mean
            elif self.params["norm_mode"] == 'min_max':
                rang = self.X_max - self.X_min
                bias = self.X_min

            if direction == 1:
                X = (X - self.X_mean) / rang
            elif direction == -1:
                X = rang * X + bias
        else:
            if self.params["norm_mode"] == 'mean_std':
                rang = self.y_std
                bias = self.y_mean
            elif self.params["norm_mode"] == 'min_max':
                rang = self.y_max - self.y_min
                bias = self.y_min

            if direction == 1:
                X = (X - self.y_mean) / rang
            elif direction == -1:
                X = rang * X + bias    

        return X

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
        dims = (-2, -1) if self.params["dimensions"] == 2 else (-3, -2, -1)
        wvs = torch.fft.fftfreq(n) # wavenumbers
        rwvs = torch.fft.rfftfreq(n) # wavenumbers of real-to-half-complex dim
        # define wavevector magnintudes
        wvs2d = torch.sqrt(wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvs3d = torch.sqrt(wvs.view(-1, 1, 1) ** 2 +
                           wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvms = wvs2d if self.params["dimensions"] == 2 else wvs3d
        kmax = wvms.max() # maximum wavevector
        beta = np.sqrt(2.0) / 3.0 # truncation factor
        # apply truncation mask
        mask = wvms > beta * kmax
        mask = mask.squeeze(0)
        fout = torch.fft.rfftn(x, dim=dims)
        fout[..., mask] = 0.0
        out = torch.fft.irfftn(fout, dim=dims)


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
        # dims = (-2, -1) if self.params["dimensions"] == 2 else (-3, -2, -1)
        dims = (-3, -2, -1)
        fy = torch.fft.rfftn(y, dim=dims) # forward real-to-half-complex FFT
        wvs = torch.fft.fftfreq(n) # wavenumbers
        rwvs = torch.fft.rfftfreq(n) # wavenumbers of real-to-half-complex dim
        # wavevector magnitudes
        wvs2d = torch.sqrt(wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvs3d = torch.sqrt(wvs.view(-1, 1, 1) ** 2 +
                           wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvms = wvs2d if self.params["dimensions"] == 2 else wvs3d
        wvmax = torch.max(wvms) # maximum wavevector magnitude
        mask = wvms > self.params["alpha"] * wvmax # define filter mask
        fy[..., mask] = 0.0 # apply filter
        y = torch.fft.irfftn(fy, dim=dims) # inverse half-complex-to-real FFT

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

        device = X.device
        
        # batch of wavevectors
        k = torch.stack([k1, k2, k3])
        k = k.to(device)
        # batch of unit vectors
        kappa = torch.stack([kappa1, kappa2, kappa3])
        kappa = kappa.to(device)
                
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
        
        fX = torch.fft.rfftn(X, dim=self.dims, norm='backward') 

        device = fX.device
        hplus = self.hplus.unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        hplus = hplus.to(device)
        hminus = self.hminus.unsqueeze(0).expand(
            X.shape[0], -1, -1, -1, -1)
        hminus = hminus.to(device)
        
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
        
        aplus = torch.einsum('bi...,bi...->b...',
                             (hminus), fX).unsqueeze(1)
        aminus = torch.einsum('bi...,bi...->b...', 
                              (hplus), fX).unsqueeze(1)

        apm = torch.cat((aplus.real, aplus.imag,
                         aminus.real, aminus.imag), dim=1)

        apm = torch.fft.irfftn(apm, dim=self.dims, norm='backward')
        
        return apm


    def from_helical(self, apm):

        apm = torch.fft.rfftn(apm, dim=self.dims, norm='backward')
        
        aplus = apm[:, 0, :, :, :].unsqueeze(1) + 1j * apm[:, 1, :, :, :].unsqueeze(1)
        aminus = apm[:, 2, :, :, :].unsqueeze(1) + 1j * apm[:, 3, :, :, :].unsqueeze(1)

        device = apm.device
        hplus = self.hplus.unsqueeze(0).expand(apm.shape[0], -1, -1, -1, -1)
        hplus = hplus.to(device)
        hminus = self.hminus.unsqueeze(0).expand(
            apm.shape[0], -1, -1, -1, -1)
        hminus = hminus.to(device)
        
        apm = aplus * hplus + aminus * hminus

        X = torch.fft.irfftn(apm, dim=self.dims, norm='backward')
        
        return X

    def divergence(self, X):

        k, _ = self.wave_vectors(X)
        k = k.unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        dims = (-3, -2, -1)
        fX = torch.fft.rfftn(X, dim=dims, norm='backward') 
        div = torch.einsum('bi...,bi...->b...', 1j * k, fX)
        div = torch.fft.irfftn(div, dim=dims, norm='backward')
        div = div.abs().max()
        
        return div

    def vorticity(self, X):

        k, _ = self.wave_vectors(X)
        k = k.unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        dims = (-3, -2, -1)
        fX = torch.fft.rfftn(X, dim=dims, norm='backward') # forward real-to-half-complex FF
        w = torch.cross(1j * k, fX, dim=1)
        w = torch.fft.irfftn(w, dim=dims, norm='backward')
        
        return w
    
def get_dataset(filenames, params):

    # === Get Dataset === #
    train_dataset = TurbDataset(filenames, params)

    return train_dataset

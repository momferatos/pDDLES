
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
            
        y = torch.from_numpy(y).transpose(0, -1)
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
                # y = y.to(g.device)
                X = self.LES_filter(y)

                if self.params["prediction_mode"] == 'large_to_small':
                    y = y - X

                X_mean += X.flatten().sum()
                y_mean += y.flatten().sum()
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

                X_std += torch.flatten((X - X_mean) ** 2).sum()
                y_std += torch.flatten((y - y_mean) ** 2).sum()
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
        dims = (-2, -1) if self.params["dimensions"] == 2 else (-3, -2, -1)
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

    def to_helical(self, X):

        n = X.shape[-2] # get DNS square linear resolution
        dims = (-3, -2, -1)
        fX = torch.fft.rfftn(X, dim=dims) # forward real-to-half-complex FFT
        wvs = torch.fft.fftfreq(n) # wavenumbers
        rwvs = torch.fft.rfftfreq(n) # wavenumbers of real-to-half-complex dim

        # wavevectors
        k1, k2, k3 = torch.meshgrid([wvs, wvs, rwvs], indexing='ij')
        kmag = torch.sqrt(k1**2 + k2**2 + k3**2)
        kappa1 = k1 / kmag
        kappa2 = k2 / kmag
        kappa3 = k3 / kmag

        # batch of wavevectors
        k = torch.stack([k1, k2, k3]).unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)

        # batch of unit vectors
        kappa = torch.stack([kappa1, kappa2, kappa3]).unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        kappasq = kappa * kappa

        return X
        #aplus = torch.fft.irfftn(aplus, dim=dims)
        #aminus = torch.fft.irfftn(aminus, dim=dims)

        #return aplus, aminus
    
def get_dataset(filenames, params):

    # === Get Dataset === #
    train_dataset = TurbDataset(filenames, params)

    return train_dataset

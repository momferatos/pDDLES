# Copyright (c) 2022-2026 Georgios Momferatos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

from __future__ import annotations

import argparse
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pathlib
import os
import sys
import shutil

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

    def __init__(self, filenames: list[str], args: argparse.Namespace) -> None:
        self.filenames = filenames # HDF5 filenames' list
        self.args = args
        self.eps = 1.0e-5
        self.device = args.device
        self.dims = (-3, -2, -1)
        self.datadict = {}
        self.indices = []

        self.filesize = self.count_file_size()
        self.datalist = []
        if self.args.noload and self.args.dev == 'gpu' and args.copy:
            if self.filenames:
                self.copy()

        return

    def _helical_basis(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Helical unit vectors h+/h- on self.device (built on first use).

        Lazy so that data-loading instances carry no device tensors:
        DataLoader workers receive a pickled copy of this dataset and only
        need CPU-side loading/truncation (see __getstate__).
        """
        cache = getattr(self, '_helical_cache', None)
        if cache is not None:
            return cache

        k, kappa = self.wave_vectors(self.args.n)

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
        mag_ezxk = torch.sqrt(mag)
        mag = torch.linalg.vecdot(kxezxk, kxezxk, dim=0)
        mag_kxezxk = torch.sqrt(mag)
        mask = (mag_ezxk != 0.0)
        sqrt2 = np.sqrt(2.0)

        hplus = (torch.where(mask, ezxk /
                             (sqrt2 * mag_ezxk) + 1j *
                             kxezxk / (sqrt2 * mag_kxezxk),
                             (ex + 1j * ey) / sqrt2))

        hminus = (torch.where(mask, ezxk /
                              (sqrt2 * mag_ezxk) - 1j *
                              kxezxk / (sqrt2 * mag_kxezxk),
                              (ex - 1j * ey) / sqrt2))

        self._helical_cache = (hplus.to(self.device),
                               hminus.to(self.device))
        return self._helical_cache

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader workers pickle the dataset: strip the lazily built
        # spectral caches so no device tensors cross process boundaries;
        # they rebuild on first use wherever they are next needed.
        state = self.__dict__.copy()
        for key in ('_helical_cache', '_wave_vectors_cache', '_wvms_cache',
                    '_trunc_mask_cache', '_les_mask_cache'):
            state.pop(key, None)
        return state


    def __len__(self) -> int:
        # The true file count. Per-rank shard sizing is TurbSampler's job;
        # dividing by world_size here shrank every sampler-less DataLoader
        # (validation, scaler fit) to the first 1/world_size of its files.
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:


        try:
            y = self.datadict[idx]
        except(KeyError):
            # print(self.args.rank, idx)
            filename = self.filenames[idx]
            y = self.filename_to_tensor(filename)

        return y


    def _wave_magnitudes(self, n: int) -> torch.Tensor:
        """Wavevector-magnitude grid for resolution n (cached on CPU).

        Depends only on n and args.dimensions, both fixed, so build it once
        and reuse instead of rebuilding fftfreq grids on every call.
        """
        cache = getattr(self, '_wvms_cache', None)
        if cache is not None and cache[0] == n:
            return cache[1]
        wvs = torch.fft.fftfreq(n)
        rwvs = torch.fft.rfftfreq(n)
        wvs2d = torch.sqrt(wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvs3d = torch.sqrt(wvs.view(-1, 1, 1) ** 2 +
                           wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvms = wvs2d if self.args.dimensions == 2 else wvs3d
        self._wvms_cache = (n, wvms)
        return wvms

    def _truncation_mask(self, n: int, device: torch.device) -> torch.Tensor:
        """Patterson-Orszag dealiasing mask (cached per n and device)."""
        cache = getattr(self, '_trunc_mask_cache', None)
        if cache is not None and cache[0] == n and cache[1] == device:
            return cache[2]
        wvms = self._wave_magnitudes(n)
        beta = np.sqrt(2.0) / 3.0 # truncation factor
        mask = (wvms > beta * wvms.max()).squeeze(0).to(device)
        self._trunc_mask_cache = (n, device, mask)
        return mask

    def _les_filter_mask(self, n: int, device: torch.device) -> torch.Tensor:
        """Sharp-cutoff LES filter mask (cached per n and device)."""
        cache = getattr(self, '_les_mask_cache', None)
        if cache is not None and cache[0] == n and cache[1] == device:
            return cache[2]
        wvms = self._wave_magnitudes(n)
        mask = (wvms > self.args.alpha * torch.max(wvms)).to(device)
        self._les_mask_cache = (n, device, mask)
        return mask

    def truncate(self, x: torch.Tensor) -> torch.Tensor:
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
        fout = torch.fft.rfftn(x, dim=self.dims, norm='ortho')
        mask = self._truncation_mask(n, fout.device) # cached dealiasing mask
        fout[..., mask] = 0.0
        out = torch.fft.irfftn(fout, dim=self.dims, norm='ortho')

        return out


    def LES_filter(self, y: torch.Tensor) -> torch.Tensor:
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
        # forward real-to-half-complex FFT
        fy = torch.fft.rfftn(y, dim=self.dims, norm='ortho')
        mask = self._les_filter_mask(n, fy.device) # cached filter mask
        fy[..., mask] = 0.0 # apply filter
        # inverse half-complex-to-real FFT
        y = torch.fft.irfftn(fy, dim=self.dims, norm='ortho')

        return y

    def wave_vectors(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:

        # k and kappa depend only on n and device, both fixed: build once.
        cache = getattr(self, '_wave_vectors_cache', None)
        if cache is not None and cache[0] == n:
            return cache[1], cache[2]

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
        self._wave_vectors_cache = (n, k, kappa)
        return k, kappa

    def helical_checks(self, X: torch.Tensor) -> None:

        k, kappa = self.wave_vectors(self.args.n)
        hplus, hminus = self._helical_basis()

        d00 = torch.linalg.vecdot(
                           hplus,
                           torch.conj_physical(hplus), dim=0)

        d00 = d00.abs().max()

        d01 = torch.linalg.vecdot(
                           hplus,
                           torch.conj_physical(hminus), dim=0)
        d01 = d01
        d01 = d01.abs().max()

        d10 = torch.linalg.vecdot(
                           hminus,
                           torch.conj_physical(hplus), dim=0)
        d10 = d10
        d10 = d10.abs().max()

        d11 = torch.linalg.vecdot(
                           hminus,
                           torch.conj_physical(hminus), dim=0)
        mask = (d11 != 0.0)
        d11 = d11.abs().max()

        print('delta: ', torch.Tensor([[d00, d01], [d10, d11]]))

        X = X.to(self.device)
        X_hel = self.to_helical(X)
        X_tr = self.from_helical(X_hel)
        diff = X_tr - X
        print('diff: ', torch.max(torch.abs(diff)))
        print('div: ', torch.max(torch.abs(self.divergence(X_tr))))

        sys.exit(0)

        return

    def to_helical(self, u: torch.Tensor,
                   outdomain: str = 'physical') -> torch.Tensor:

        fu = torch.fft.rfftn(u, dim=self.dims, norm='ortho')

        hplus, hminus = self._helical_basis()
        hplus = hplus.unsqueeze(0).expand(u.shape[0], -1, -1, -1, -1)
        hminus = hminus.unsqueeze(0).expand(
            u.shape[0], -1, -1, -1, -1)

        fuplus = torch.einsum('bi...,bi...->b...',
                              fu, torch.conj_physical(hplus)).unsqueeze(1)
        fuminus = torch.einsum('bi...,bi...->b...',
                               fu, torch.conj_physical(hminus)).unsqueeze(1)

        if outdomain == 'physical':
            uplus = torch.fft.irfftn(fuplus, dim=self.dims, norm='ortho')
            uminus = torch.fft.irfftn(fuminus, dim=self.dims, norm='ortho')
            out = torch.cat((uplus, uminus), dim=1)
        elif outdomain == 'fourier':
            out = torch.cat((fuplus, fuminus), dim=1)

        return out


    def from_helical(self, fupm: torch.Tensor,
                     indomain: str = 'physical') -> torch.Tensor:

        if indomain == 'physical':
            uplus = fupm[:, 0, :, :, :].unsqueeze(1)
            uminus = fupm[:, 1, :, :, :].unsqueeze(1)
            fuplus = torch.fft.rfftn(uplus, dim=self.dims, norm='ortho')
            fuminus = torch.fft.rfftn(uminus, dim=self.dims, norm='ortho')
        elif indomain == 'fourier':
            fuplus = fupm[:, 0, :, :, :].unsqueeze(1)
            fuminus = fupm[:, 1, :, :, :].unsqueeze(1)

        hplus, hminus = self._helical_basis()
        hplus = hplus.unsqueeze(0).expand(fupm.shape[0], -1, -1, -1, -1)

        hminus = hminus.unsqueeze(0).expand(
            fupm.shape[0], -1, -1, -1, -1)


        fu = fuplus * hplus + fuminus * hminus

        u = torch.fft.irfftn(fu, dim=self.dims, norm='ortho')

        return u

    def divergence(self, X: torch.Tensor) -> torch.Tensor:

        k, _ = self.wave_vectors(self.args.n)

        k = k.unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        dims = (-3, -2, -1)
        fX = torch.fft.rfftn(X, dim=self.dims, norm='ortho')
        div = torch.linalg.vecdot(1j * k, torch.conj_physical(fX), dim=1)
        div = torch.fft.irfftn(div, dim=self.dims, norm='ortho')
        div = torch.mean(torch.abs(div))

        return div

    def longitudinal_gradients(self, X: torch.Tensor) -> torch.Tensor:

        k, _ = self.wave_vectors(self.args.n)

        k = k.unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        dims = (-3, -2, -1)
        fX = torch.fft.rfftn(X, dim=self.dims, norm='ortho')
        lgrads = -1j * k * fX
        lgrads= torch.fft.irfftn(lgrads, dim=self.dims, norm='ortho')

        lgrads = lgrads.flatten()

        return lgrads

    def vorticity(self, X: torch.Tensor) -> torch.Tensor:

        k, _ = self.wave_vectors(self.args.n)

        k = k.unsqueeze(0).expand(X.shape[0], -1, -1, -1, -1)
        dims = (-3, -2, -1)
        # forward real-to-half-complex FF
        fX = torch.fft.rfftn(X, dim=self.dims, norm='ortho')
        w = torch.cross(1j * k, fX, dim=1)
        w = torch.fft.irfftn(w, dim=self.dims, norm='ortho')

        return w

    def subgrid_scale_tensor(self, y: torch.Tensor) -> torch.Tensor:


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

    def filename_to_tensor(self, filename: str) -> torch.Tensor:
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
                # Add extra tensor dimension required by PyTorch
                y = y.unsqueeze(0)

        return y

    def count_file_size(self) -> float:


        if not self.filenames:
            return 0.0

        y = self.filename_to_tensor(self.filenames[0])

        size = sys.getsizeof(y.storage)
        size *= len(self.filenames)
        GB = 1024.0 ** 3
        size /= GB

        return size

    def copy(self) -> None:

        if not self.filenames or self.args.localrank !=0:
            return

        dest_dir = os.environ['LOCALSCRATCH']
        dest_dir = os.path.join(dest_dir, 'tmp')
        pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

        filenames = []
        nfiles = len(self.filenames)
        for nfile, src_filename in enumerate(self.filenames):
            filename = os.path.split(src_filename)[-1]
            dest_filename = os.path.join(dest_dir, filename)
            if nfile % 250 == 0:
                print(f'Copying file {nfile}/{nfiles}...')
            if not os.path.exists(dest_filename):
                try:
                    shutil.copyfile(src_filename, dest_filename)
                    filenames.append(dest_filename)
                except(IOError):
                    filenames.append(src_filename)


        self.filenames = filenames

        return

    def load(self, indices: list[int]) -> None:
        if self.args.noload:
            return
        self.indices = indices
        nfiles = len(self.indices)
        for nfile, idx in enumerate(self.indices):
            filename = self.filenames[idx]
            if nfile % 100 == 0:
                print(f'Loading file {nfile}/{nfiles} to memory.')
            y = self.filename_to_tensor(filename)
            self.datadict[idx] = y
        return

_helper_cache = {}

def get_helper(args: argparse.Namespace) -> TurbDataset:
    """Shared file-less TurbDataset for the spectral utilities.

    Models, blocks, losses and scalers only need the transform methods
    (LES_filter, to/from_helical, divergence, ...), whose precomputed
    helical basis (hplus/hminus/k/kappa, ~n^3-sized tensors on
    args.device) depends only on the key below. A private instance per
    FourierBlock/WaveletBlock duplicated that basis once per block.
    """
    key = (int(args.n), str(args.device), int(args.dimensions),
           float(args.alpha))
    if key not in _helper_cache:
        _helper_cache[key] = TurbDataset([], args)
    return _helper_cache[key]


def get_dataset(filenames: list[str],
                args: argparse.Namespace) -> TurbDataset:

    # === Get Dataset === #
    train_dataset = TurbDataset(filenames, args)

    return train_dataset

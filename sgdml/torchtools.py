#!/usr/bin/python

# MIT License
#
# Copyright (c) 2019-2020 Jan Hermann, Stefan Chmiela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.desc import Desc


def expand_tril(xs):
    n = int((1 + np.sqrt(8 * xs.shape[-1] + 1)) / 2)
    xs_full = torch.zeros((xs.shape[0], n, n), dtype=xs.dtype, device=xs.device)
    i, j = np.tril_indices(n, k=-1)
    xs_full[:, i, j] = xs
    xs_full[:, j, i] = xs

    # needed?
    #i, j = np.diag_indices(n)
    #xs_full[:, i, j] = 0

    return xs_full

def expand_tril_multiply(xs, mat):
    
    n = mat.shape[1]
    i, j = np.tril_indices(n, k=-1)
    mat[:, i, j, :] *= xs[:, :, None]
    mat[:, j, i, :] *= xs[:, :, None]

    return mat


class GDMLTorchPredict(nn.Module):
    """
    PyTorch version of :class:`~predict.GDMLPredict`. Derives from
    :class:`torch.nn.Module`. Contains no trainable parameters.
    """

    def __init__(self, model, lat_and_inv=None, batch_size=None, max_memory=None):
        """
        Parameters
        ----------
        model : Mapping
            Obtained from :meth:`~train.GDMLTrain.train`.
        lat_and_inv : tuple of :obj:`numpy.ndarray`
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
        batch_size : int, optional
            Maximum batch size of geometries for prediction. Calculated from
            :paramref:`max_memory` if not given.
        max_memory : float, optional
            (unit GB) Maximum allocated memory for prediction.
        """

        global _batch_size

        super(GDMLTorchPredict, self).__init__()

        self._log = logging.getLogger(__name__)

        model = dict(model)

        self._dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._lat_and_inv = (
            None
            if lat_and_inv is None
            else (
                torch.tensor(lat_and_inv[0], device=self._dev),
                torch.tensor(lat_and_inv[1], device=self._dev),
            )
        )

        self._sig = int(model['sig'])
        self._c = float(model['c'])
        self._std = float(model.get('std', 1))

        desc_siz = model['R_desc'].shape[0]
        n_perms, self._n_atoms = model['perms'].shape
        perm_idxs = (
            torch.tensor(model['tril_perms_lin'], device=self._dev)
            .view(-1, n_perms)
            .t()
        )
        self._xs_train, self._Jx_alphas = (
            nn.Parameter(
                xs.repeat(1, n_perms)[:, perm_idxs].reshape(-1, desc_siz),
                requires_grad=False,
            )
            for xs in (
                torch.tensor(model['R_desc'], device=self._dev).t(),
                torch.tensor(np.array(model['R_d_desc_alpha']), device=self._dev),
            )
        )

        if max_memory is None:
            if torch.cuda.is_available():
                max_memory = max(
                    [
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count())
                    ]
                )
            else:
                max_memory = 32 # TODO: 32 GB as default (hardcoded for now...)

        _batch_size = (
            int(2 ** 30 * max_memory) // self._memory_per_sample() # max_memory: GB to bytes
            if batch_size is None
            else batch_size
        )
        if torch.cuda.is_available():
            _batch_size *= torch.cuda.device_count()

        n_atoms = model['z'].shape[0]
        self.desc = Desc(n_atoms) # NOTE: max processes not set!!

        self.perm_idxs = perm_idxs
        self.n_perms = n_perms

    def set_alphas(self, R_d_desc, alphas):
        """
        Reconfigure the current model with a new set of regression parameters.
        This is necessary when training the model iteratively.

        Parameters
        ----------
                R_d_desc : :obj:`numpy.ndarray`
                    Array containing the Jacobian of the descriptor for
                    each training point.
                alphas : :obj:`numpy.ndarray`
                    1D array containing the new model parameters.
        """

        dim_d = self.desc.dim
        dim_i = self.desc.dim_i

        R_d_desc_alpha = self.desc.d_desc_dot_vec(R_d_desc, alphas.reshape(-1, dim_i))

        #r_dim = R_d_desc.shape[2]
        #R_d_desc_alpha = np.einsum('jik,jk->ji', R_d_desc, alphas.reshape(-1, r_dim))

        xs = torch.from_numpy(R_d_desc_alpha).to(self._dev)

        self._Jx_alphas = nn.Parameter(
            xs.repeat(1, self.n_perms)[:, self.perm_idxs].reshape(-1, dim_d),
            requires_grad=False,
        )

    def _memory_per_sample(self):
        return 3 * self._xs_train.nelement() * self._xs_train.element_size()

    def _batch_size(self):
        return _batch_size

    def _forward(self, Rs):
        sig = self._sig
        q = np.sqrt(5) / sig

        diffs = Rs[:, :, None, :] - Rs[:, None, :, :]
        if self._lat_and_inv is not None:
            diffs_shape = diffs.shape
            #diffs = self.desc.pbc_diff(diffs.reshape(-1, 3), self._lat_and_inv).reshape(
            #    diffs_shape
            #)

            lat, lat_inv = self._lat_and_inv

            if lat.device != Rs.device:
                lat = lat.to(Rs.device)
                lat_inv = lat_inv.to(Rs.device)
            
            diffs = diffs.reshape(-1, 3)

            c = lat_inv.mm(diffs.t())
            diffs -= lat.mm(c.round()).t()

            diffs = diffs.reshape(diffs_shape)

        dists = diffs.norm(dim=-1)
        i, j = np.diag_indices(self._n_atoms)
        dists[:, i, j] = np.inf

        i, j = np.tril_indices(self._n_atoms, k=-1)
        xs = 1 / dists[:, i, j]  # R_desc (1000, 36)

        x_diffs = (q * xs)[:, None, :] - q * self._xs_train
        x_dists = x_diffs.norm(dim=-1)
        exp_xs = 5.0 / (3 * sig ** 2) * torch.exp(-x_dists)
        dot_x_diff_Jx_alphas = (x_diffs * self._Jx_alphas).sum(dim=-1)
        exp_xs_1_x_dists = exp_xs * (1 + x_dists)
        F1s_x = ((exp_xs * dot_x_diff_Jx_alphas)[..., None] * x_diffs).sum(dim=1)
        F2s_x = exp_xs_1_x_dists.mm(self._Jx_alphas)
        #Fs_x = (F1s_x - F2s_x)  * self._std

        #Fs = ((expand_tril(Fs_x) / dists ** 3)[..., None] * diffs).sum(
        #    dim=1
        #)  # * R_d_desc

        #Fs = (expand_tril(Fs_x * (xs ** 3))[..., None] * diffs).sum(
        #    dim=1
        #)  # * R_d_desc

        #Fs = expand_tril_multiply(Fs_x * (xs ** 3), diffs).sum(
        #    dim=1
        #)

        Fs_x = (F1s_x - F2s_x) * (xs ** 3)
        diffs[:, i, j, :] *= Fs_x[..., None]
        diffs[:, j, i, :] *= Fs_x[..., None]

        Fs = diffs.sum(dim=1) * self._std

        Es = (exp_xs_1_x_dists * dot_x_diff_Jx_alphas).sum(dim=-1) / q
        Es = Es * self._std + self._c

        return Es, Fs

    def forward(self, Rs):
        """
        Predict energy and forces for a batch of geometries.

        Parameters
        ----------
        Rs : :obj:`torch.Tensor`
            (dims M x N x 3) Cartesian coordinates of M molecules composed of N atoms

        Returns
        -------
        E : :obj:`torch.Tensor`
            (dims M) Molecular energies
        F : :obj:`torch.Tensor`
            (dims M x N x 3) Nuclear gradients of the energy
        """

        global _batch_size

        assert Rs.dim() == 3
        assert Rs.shape[1:] == (self._n_atoms, 3)

        dtype = Rs.dtype
        Rs = Rs.double()

        retry = True
        while retry:

            try:
                Es, Fs = zip(*map(self._forward, DataLoader(Rs, batch_size=_batch_size)))
                retry = False

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if _batch_size > 2:

                        import gc
                        gc.collect()

                        torch.cuda.empty_cache()

                        _batch_size -= 1
                        retry = True

                    else:
                        self._log.critical(
                            'Could not allocate enough memory to evaluate model, despite reducing batch size.'
                        )
                        print()
                        sys.exit()
                else:
                    raise e

        return torch.cat(Es).to(dtype), torch.cat(Fs).to(dtype)

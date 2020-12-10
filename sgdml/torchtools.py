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

import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.desc import Desc


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

        self.n_atoms = model['z'].shape[0]

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

        # constant memory requirement (bytes): _xs_train and _Jx_alphas
        const_memory = 2 * self._xs_train.nelement() * self._xs_train.element_size()

        if max_memory is None:
            if torch.cuda.is_available():
                max_memory = min(
                    [
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count())
                    ]
                )
            else:
                max_memory = int(
                    2 ** 30 * 32
                )  # 32 GB to bytes as default (hardcoded for now...)

        _batch_size = (
            max((max_memory - const_memory) // self._memory_per_sample(), 1)
            if batch_size is None
            else batch_size
        )
        if torch.cuda.is_available():
            _batch_size *= torch.cuda.device_count()

        self.desc = Desc(self.n_atoms)  # NOTE: max processes not set!!

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
        xs = torch.from_numpy(R_d_desc_alpha).to(self._dev)

        self._Jx_alphas = nn.Parameter(
            xs.repeat(1, self.n_perms)[:, self.perm_idxs].reshape(-1, dim_d),
            requires_grad=False,
        )

    def _memory_per_sample(self):
        # return 3 * self._xs_train.nelement() * self._xs_train.element_size() # size of 'diffs' (biggest object in 'forward')

        # peak memory:
        # N * a * a * 3
        # N * d * 2
        # N * n_perms*N_train * (d+4)

        dim_d = self._xs_train.shape[1]

        total = (dim_d * 2 + self.n_atoms) * 3
        total += dim_d * 2
        total += self._xs_train.shape[0] * (dim_d + 4)

        return total * self._xs_train.element_size()

    def _batch_size(self):
        return _batch_size

    def _forward(self, Rs):

        sig = self._sig
        q = np.sqrt(5) / sig

        diffs = Rs[:, :, None, :] - Rs[:, None, :, :]  # N, a, a, 3
        if self._lat_and_inv is not None:
            diffs_shape = diffs.shape
            # diffs = self.desc.pbc_diff(diffs.reshape(-1, 3), self._lat_and_inv).reshape(
            #    diffs_shape
            # )

            lat, lat_inv = self._lat_and_inv

            if lat.device != Rs.device:
                lat = lat.to(Rs.device)
                lat_inv = lat_inv.to(Rs.device)

            diffs = diffs.reshape(-1, 3)

            c = lat_inv.mm(diffs.t())
            diffs -= lat.mm(c.round()).t()

            diffs = diffs.reshape(diffs_shape)

        dists = diffs.norm(dim=-1)  # N, a, a

        # i, j = np.diag_indices(self._n_atoms)
        # dists[:, i, j] = np.inf

        i, j = np.tril_indices(self._n_atoms, k=-1)
        xs = 1 / dists[:, i, j]  # R_desc # N, d

        # current:
        # diffs: N, a, a, 3
        # dists: N, a, a
        # xs: # N, d

        del dists

        # current:
        # diffs: N, a, a, 3
        # xs: # N, d

        x_diffs = (q * xs)[:, None, :] - q * self._xs_train  # N, n_perms*N_train, d
        x_dists = x_diffs.norm(dim=-1)  # N, n_perms*N

        exp_xs = 5.0 / (3 * sig ** 2) * torch.exp(-x_dists)  # N, n_perms*N_train

        # dot_x_diff_Jx_alphas = (x_diffs * self._Jx_alphas).sum(dim=-1)
        dot_x_diff_Jx_alphas = torch.einsum(
            'ijk,jk->ij', x_diffs, self._Jx_alphas
        )  # N, n_perms*N_train
        exp_xs_1_x_dists = exp_xs * (1 + x_dists)  # N, n_perms*N_train

        # F1s_x = ((exp_xs * dot_x_diff_Jx_alphas)[..., None] * x_diffs).sum(dim=1)
        # F2s_x = exp_xs_1_x_dists.mm(self._Jx_alphas)

        # Fs_x = ((exp_xs * dot_x_diff_Jx_alphas)[..., None] * x_diffs).sum(dim=1)
        Fs_x = torch.einsum(
            'ij,ij,ijk->ik', exp_xs, dot_x_diff_Jx_alphas, x_diffs
        )  # N, d

        # current:
        # diffs: N, a, a, 3
        # xs: # N, d
        # x_diffs: # N, n_perms*N_train, d
        # x_dists: # N, n_perms*N_train
        # exp_xs: # N, n_perms*N_train
        # dot_x_diff_Jx_alphas: N, n_perms*N_train
        # exp_xs_1_x_dists: N, n_perms*N_train
        # Fs_x: N, d

        del exp_xs
        del x_diffs

        Fs_x -= exp_xs_1_x_dists.mm(self._Jx_alphas)  # N, d

        # current:
        # diffs: N, a, a, 3
        # xs: # N, d
        # x_dists: # N, n_perms*N
        # dot_x_diff_Jx_alphas: N, n_perms*N
        # exp_xs_1_x_dists: N, n_perms*N
        # Fs_x: N, d

        # Fs_x = (F1s_x - F2s_x) * (xs ** 3)
        Fs_x *= xs ** 3
        diffs[:, i, j, :] *= Fs_x[..., None]
        diffs[:, j, i, :] *= Fs_x[..., None]

        Fs = diffs.sum(dim=1) * self._std

        del diffs

        # Es = (exp_xs_1_x_dists * dot_x_diff_Jx_alphas).sum(dim=-1) / q
        Es = torch.einsum('ij,ij->i', exp_xs_1_x_dists, dot_x_diff_Jx_alphas) / q
        Es *= self._std
        Es += self._c

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

        while True:
            try:
                Es, Fs = zip(
                    *map(self._forward, DataLoader(Rs, batch_size=_batch_size))
                )
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if _batch_size > (torch.cuda.device_count() + 1):

                        import gc
                        gc.collect()

                        torch.cuda.empty_cache()

                        _batch_size -= 1

                    else:
                        self._log.critical(
                            'Could not allocate enough memory to evaluate model, despite reducing batch size.'
                        )
                        print()
                        sys.exit()
                else:
                    raise e
            else:
                break

        return torch.cat(Es).to(dtype), torch.cat(Fs).to(dtype)

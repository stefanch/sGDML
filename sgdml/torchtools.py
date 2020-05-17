#!/usr/bin/python

# MIT License
#
# Copyright (c) 2019 Jan Hermann
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import desc


def expand_tril(xs):
    n = int((1 + np.sqrt(8 * xs.shape[-1] + 1)) / 2)
    xs_full = torch.zeros((xs.shape[0], n, n), dtype=xs.dtype, device=xs.device)
    i, j = np.tril_indices(n, k=-1)
    xs_full[:, i, j] = xs
    xs_full[:, j, i] = xs
    i, j = np.diag_indices(n)
    xs_full[:, i, j] = 0
    return xs_full


class GDMLTorchPredict(nn.Module):
    """
    PyTorch version of :class:`~predict.GDMLPredict`. Derives from
    :class:`torch.nn.Module`. Contains no trainable parameters.
    """

    def __init__(self, model, lat_and_inv=None, batch_size=None, max_memory=1.0):
        """
        Parameters
        ----------
        model : Mapping
            Obtained from :meth:`~train.GDMLTrain.train`.
        batch_size : int, optional
            Maximum batch size of geometries for prediction. Calculated from
            :paramref:`max_memory` if not given.
        lat_and_inv : tuple of :obj:`numpy.ndarray`
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
        max_memory : float, optional
            (unit GB) Maximum allocated memory for prediction.
        """

        super(GDMLTorchPredict, self).__init__()

        model = dict(model)  # hack

        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._lat_and_inv = (
            None
            if lat_and_inv is None
            else (
                torch.tensor(lat_and_inv[0], device=torch_device),
                torch.tensor(lat_and_inv[1], device=torch_device),
            )
        )

        self._batch_size = batch_size
        self._max_memory = int(2 ** 30 * max_memory)
        self._sig = int(model['sig'])
        self._c = float(model['c'])
        self._std = float(model.get('std', 1))
        self.use_descriptor = model['use_descriptor']

        desc_siz = model['R_desc'].shape[0]
        n_perms, self._n_atoms = model['perms'].shape
        perm_idxs = torch.tensor(model['tril_perms_lin']).view(-1, n_perms).t()
        self._xs_train, self._Jx_alphas = (
            nn.Parameter(
                xs.repeat(1, n_perms)[:, perm_idxs].reshape(-1, desc_siz),
                requires_grad=False,
            )
            for xs in (
                torch.tensor(model['R_desc']).t(),
                torch.tensor(np.array(model['R_d_desc_alpha'])),
            )
        )

    def _memory_per_sample(self):
        return 3 * self._xs_train.nelement() * self._xs_train.element_size()

    def _forward(self, Rs):
        sig = self._sig
        q = np.sqrt(5) / sig

        diffs = Rs[:, :, None, :] - Rs[:, None, :, :]

        if self._lat_and_inv is not None:
            diffs_shape = diffs.shape
            diffs = desc.pbc_diff_torch(
                diffs.reshape(-1, 3), self._lat_and_inv
            ).reshape(diffs_shape)

        dists = diffs.norm(dim=-1)
        i, j = np.diag_indices(self._n_atoms)

        dists[:, i, j] = np.inf
        i, j = np.tril_indices(self._n_atoms, k=-1)

        if self.use_descriptor[0] == 'coulomb_matrix':
            xs = 1 / dists[:, i, j]
        elif self.use_descriptor[0] == 'exp_decay_matrix':
            xs = torch.exp(-dists[:, i, j])

        x_diffs = (q * xs)[:, None, :] - q * self._xs_train
        x_dists = x_diffs.norm(dim=-1)
        exp_xs = 5.0 / (3 * sig ** 2) * torch.exp(-x_dists)
        dot_x_diff_Jx_alphas = (x_diffs * self._Jx_alphas).sum(dim=-1)
        exp_xs_1_x_dists = exp_xs * (1 + x_dists)
        F1s_x = ((exp_xs * dot_x_diff_Jx_alphas)[..., None] * x_diffs).sum(dim=1)
        F2s_x = exp_xs_1_x_dists.mm(self._Jx_alphas)
        Fs_x = (F1s_x - F2s_x) * self._std

        Fs = ((expand_tril(Fs_x) / dists ** 3)[..., None] * diffs).sum(dim=1)

        Es = (exp_xs_1_x_dists * dot_x_diff_Jx_alphas).sum(dim=-1) / q
        Es = self._c + Es * self._std

        return Es, Fs

    def forward(self, Rs, batch_size=None, max_memory=None):
        """
        Predict energy and forces for a batch of geometries.

        Parameters
        ----------
        R : :obj:`torch.Tensor`
            (dims M x N x 3) Cartesian coordinates of M molecules composed of N atoms

        Returns
        -------
        E : :obj:`torch.Tensor`
            (dims M) Molecular energies
        F : :obj:`torch.Tensor`
            (dims M x N x 3) Nuclear gradients of the energy
        """

        assert Rs.dim() == 3
        assert Rs.shape[1:] == (self._n_atoms, 3)

        dtype = Rs.dtype
        Rs = Rs.double()
        batch_size = self._batch_size or self._max_memory // self._memory_per_sample()

        Es, Fs = zip(*map(self._forward, DataLoader(Rs, batch_size=batch_size)))

        return torch.cat(Es).to(dtype), torch.cat(Fs).to(dtype)

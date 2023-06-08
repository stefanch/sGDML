#!/usr/bin/python

# MIT License
#
# Copyright (c) 2019-2023 Stefan Chmiela, Jan Hermann
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

import os
import sys
import logging
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    _torch_mps_is_available = torch.backends.mps.is_available()
except AttributeError:
    _torch_mps_is_available = False
_torch_mps_is_available = False

try:
    _torch_cuda_is_available = torch.cuda.is_available()
except AttributeError:
    _torch_cuda_is_available = False


from .utils.desc import Desc
from .utils import ui

_dtype = torch.float64


def _next_batch_size(n_total, batch_size):

    batch_size += 1
    while n_total % batch_size != 0:
        batch_size += 1

    return batch_size


class GDMLTorchAssemble(nn.Module):
    """
    PyTorch version of the kernel assembly routines in :class:`~predict.GDMLTrain`.
    Derives from :class:`torch.nn.Module`. Contains no trainable parameters.
    """

    def __init__(
        self,
        J,
        tril_perms_lin,
        sig,
        use_E_cstr,
        R_desc_torch,
        R_d_desc_torch,
        out,
        callback=None,
    ):

        global _n_batches, _n_perm_batches

        super(GDMLTorchAssemble, self).__init__()

        self._log = logging.getLogger(__name__)

        self.callback = callback

        self.n_train, self.dim_d = R_d_desc_torch.shape[:2]
        self.n_atoms = int((1 + np.sqrt(8 * self.dim_d + 1)) / 2)
        self.dim_i = 3 * self.n_atoms

        self.sig = float(sig)
        self.tril_perms_lin = tril_perms_lin
        self.n_perms = len(self.tril_perms_lin) // self.dim_d

        self.use_E_cstr = use_E_cstr

        self.R_desc_torch = nn.Parameter(R_desc_torch.type(_dtype), requires_grad=False)
        self.R_d_desc_torch = nn.Parameter(
            R_d_desc_torch.type(_dtype), requires_grad=False
        )

        self._desc = Desc(self.n_atoms)

        self.J = J
        _n_batches = 1
        _n_perm_batches = 1

        self.out = out

    def _forward(
        self,
        j,
    ):

        global _n_batches, _n_perm_batches

        if type(j) is tuple:  # selective/"fancy" indexing
            (
                K_j,
                j,
                keep_idxs_3n,
            ) = j  # (block index in final K, block index global, indices of partials within block)
            blk_j_len = len(keep_idxs_3n)
            blk_j = slice(K_j, K_j + blk_j_len)

        else:  # sequential indexing
            blk_j_len = self.dim_i
            K_j = (
                j * self.dim_i
                if j < self.n_train
                else self.n_train * self.dim_i + (j % self.n_train)
            )
            blk_j = (
                slice(K_j, K_j + self.dim_i)
                if j < self.n_train
                else slice(K_j, K_j + 1)
            )
            keep_idxs_3n = slice(None)  # same as [:]

        q = np.sqrt(5) / self.sig

        if (
            j < self.n_train
        ):  # This column only contrains second and first derivative constraints.

            # Create decompressed a 'rj_d_desc'.
            rj_d_desc_decomp_torch = self._desc.d_desc_from_comp(
                self.R_d_desc_torch[j % self.n_train, :, :]
            )[0][:, keep_idxs_3n]

            n_perms_done = 0
            for perm_batch in np.array_split(
                np.arange(self.n_perms), min(_n_perm_batches, self.n_perms)
            ):

                tril_perms_lin_batch = (
                    self.tril_perms_lin.reshape(-1, self.n_perms)[:, perm_batch]
                    - n_perms_done * self.dim_d
                ).ravel()  # index shift

                n_perms_batch = len(perm_batch)
                n_perms_done += n_perms_batch

                # Create a permutated 'rj_desc'.
                rj_desc_perms_torch = torch.reshape(
                    torch.tile(self.R_desc_torch[j, :], (n_perms_batch,))[
                        tril_perms_lin_batch
                    ],
                    (-1, n_perms_batch),
                ).T

                # Create a permutated 'rj_d_desc'.
                rj_d_desc_perms_torch = torch.reshape(
                    torch.tile(rj_d_desc_decomp_torch.T, (n_perms_batch,))[
                        :, tril_perms_lin_batch
                    ],
                    (-1, self.dim_d, n_perms_batch),
                )

                for i_batch in np.array_split(np.arange(self.n_train), _n_batches):

                    x_diffs = q * (
                        self.R_desc_torch[i_batch, None, :]
                        - rj_desc_perms_torch[None, :, :]
                    )  # N, n_perms, d

                    x_dists = x_diffs.norm(dim=-1)  # N, n_perms

                    exp_xs = torch.exp(-x_dists) * (q**2) / 3  # N, n_perms
                    exp_xs_1_x_dists = exp_xs * (1 + x_dists)  # N, n_perms*N_train

                    del x_dists  # E_cstr

                    diff_ab_outer_perms_torch = torch.einsum(
                        '...ki,...kj->...ij',  # (slow)
                        x_diffs * exp_xs[:, :, None],  # N, n_perms, d
                        torch.einsum(
                            '...ki,jik -> ...kj',
                            x_diffs,
                            rj_d_desc_perms_torch,
                        ),  # N, n_perms, a*3
                    )  # N, n_perms, a*3
                    del exp_xs

                    if not self.use_E_cstr:
                        del x_diffs

                    diff_ab_outer_perms_torch -= torch.einsum(
                        'ikj,...j->...ki',
                        rj_d_desc_perms_torch,
                        exp_xs_1_x_dists,
                    )

                    if not self.use_E_cstr:
                        del exp_xs_1_x_dists

                    R_d_desc_decomp_torch = self._desc.d_desc_from_comp(
                        self.R_d_desc_torch[i_batch, :, :]
                    )

                    k = torch.einsum(
                        '...ij,...ik->...kj',
                        diff_ab_outer_perms_torch,  # N, d, 3*a
                        R_d_desc_decomp_torch,
                    )
                    del diff_ab_outer_perms_torch
                    del R_d_desc_decomp_torch

                    blk_i = slice(
                        i_batch[0] * self.dim_i, (i_batch[-1] + 1) * self.dim_i
                    )

                    k_np = k.cpu().numpy().reshape(-1, blk_j_len)
                    if (
                        n_perms_done == n_perms_batch
                    ):  # first permutation batch iteration
                        self.out[blk_i, blk_j] = k_np
                    else:
                        self.out[blk_i, blk_j] = self.out[blk_i, blk_j] + k_np
                    del k

                    # First derivative constraints
                    if self.use_E_cstr:

                        K_fe = (x_diffs / q) * exp_xs_1_x_dists[:, :, None]
                        del x_diffs
                        del exp_xs_1_x_dists

                        K_fe = -torch.einsum(
                            '...ik,jki -> ...j', K_fe, rj_d_desc_perms_torch
                        )

                        E_off_i = self.n_train * self.dim_i
                        i_batch_off = i_batch + E_off_i
                        self.out[
                            i_batch_off[0] : (i_batch_off[-1] + 1), blk_j
                        ] = K_fe.cpu().numpy()

                del rj_desc_perms_torch
                del rj_d_desc_perms_torch

        else:

            if self.use_E_cstr:

                n_perms_done = 0
                for perm_batch in np.array_split(
                    np.arange(self.n_perms), min(_n_perm_batches, self.n_perms)
                ):

                    tril_perms_lin_batch = (
                        self.tril_perms_lin.reshape(-1, self.n_perms)[:, perm_batch]
                        - n_perms_done * self.dim_d
                    ).ravel()  # index shift

                    n_perms_batch = len(perm_batch)
                    n_perms_done += n_perms_batch

                    for i_batch in np.array_split(np.arange(self.n_train), _n_batches):

                        ri_desc_perms_torch = torch.reshape(
                            torch.tile(
                                self.R_desc_torch[i_batch, :], (1, n_perms_batch)
                            )[:, tril_perms_lin_batch],
                            (len(i_batch), -1, n_perms_batch),
                        )

                        # Create decompressed a 'ri_d_desc'.
                        ri_d_desc_decomp_torch = self._desc.d_desc_from_comp(
                            self.R_d_desc_torch[i_batch, :, :]
                        )

                        ri_d_desc_perms_torch = torch.reshape(
                            torch.tile(ri_d_desc_decomp_torch, (1, n_perms_batch, 1))[
                                :, tril_perms_lin_batch, :
                            ],
                            (len(i_batch), self.dim_d, n_perms_batch, -1),
                        )
                        # del ri_d_desc_decomp_torch

                        x_diffs = q * (
                            self.R_desc_torch[j % self.n_train, None, :, None]
                            - ri_desc_perms_torch
                        )

                        x_dists = x_diffs.norm(dim=1)

                        exp_xs = torch.exp(-x_dists) * (q**2) / 3
                        exp_xs_1_x_dists = exp_xs * (1 + x_dists)

                        K_fe = x_diffs / q * exp_xs_1_x_dists[:, None, :]
                        K_fe = -torch.einsum(
                            '...ik,...ikj -> ...j', K_fe, ri_d_desc_perms_torch
                        ).ravel()
                        k_fe = K_fe.cpu().numpy()

                        k_ee = -torch.einsum(
                            '...i,...i -> ...',
                            1 + x_dists * (1 + x_dists / 3),
                            torch.exp(-x_dists),
                        )
                        k_ee = k_ee.cpu().numpy()

                        E_off_i = (
                            self.n_train * self.dim_i
                        )  # Account for 'alloc_extra_rows'!.
                        blk_i_full = slice(
                            i_batch[0] * self.dim_i, (i_batch[-1] + 1) * self.dim_i
                        )
                        if (
                            n_perms_done == n_perms_batch
                        ):  # first permutation batch iteration
                            self.out[blk_i_full, K_j] = k_fe
                            self.out[E_off_i + i_batch, K_j] = k_ee
                        else:
                            self.out[blk_i_full, K_j] = self.out[blk_i_full, K_j] + k_fe
                            self.out[E_off_i + i_batch, K_j] = (
                                self.out[E_off_i + i_batch, K_j] + k_ee
                            )

        return blk_j.stop - blk_j.start

    def forward(self, J_indx):

        global _n_batches, _n_perm_batches

        for i in J_indx:
            while True:
                try:
                    done = self._forward(self.J[i])
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        if _torch_cuda_is_available:
                            torch.cuda.empty_cache()

                        if _n_batches < self.n_train:
                            _n_batches = _next_batch_size(self.n_train, _n_batches)

                            self._log.debug(
                                'Assembling each kernel column in {} batches, i.e. {} points/batch ({} points in total).'.format(
                                    _n_batches,
                                    self.n_train // _n_batches,
                                    self.n_train,
                                )
                            )

                        elif _n_perm_batches < self.n_perms:
                            _n_perm_batches = _next_batch_size(
                                self.n_perms, _n_perm_batches
                            )

                            self._log.debug(
                                'Generating permutations in {} batches, i.e. {} permutations/batch ({} permutations in total).'.format(
                                    _n_perm_batches,
                                    self.n_perms // _n_perm_batches,
                                    self.n_perms,
                                )
                            )

                        else:
                            self._log.critical(
                                'Could not allocate enough memory to assemble kernel matrix, even block-by-block and/or handling perms in batches.'
                            )
                            print()
                            os._exit(1)
                    else:
                        raise e
                else:
                    if self.callback is not None:
                        self.callback(done)

                    break


class GDMLTorchPredict(nn.Module):
    """
    PyTorch version of :class:`~predict.GDMLPredict`. Derives from
    :class:`torch.nn.Module`. Contains no trainable parameters.
    """

    def __init__(
        self,
        model,
        lat_and_inv=None,
        batch_size=None,
        n_perm_batches=1,
        max_memory=None,
        max_processes=None,
        log_level=None,
    ):
        """
        Parameters
        ----------
        model : Mapping
            Obtained from :meth:`~train.GDMLTrain.train`.
        lat_and_inv : tuple of :obj:`numpy.ndarray`
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
        batch_size : int, optional
            Maximum batch size of geometries for prediction. Calculated from
            :paramref:`max_mem` if not given.
        n_perm_batches : int, optional
            Divide the processing of all symmetries for each point into smaller
            batches or precompute all in the beginning (needs  more memmory, but faster)?
        max_memory : float, optional
            (unit GB) Maximum allowed CPU memory for prediction (GPU memory always unlimited)
        """

        global _batch_size, _n_perm_batches

        super(GDMLTorchPredict, self).__init__()

        self._log = logging.getLogger(__name__)
        if log_level is not None:
            self._log.setLevel(log_level)

        model = dict(model)

        self._lat_and_inv = (
            None
            if lat_and_inv is None
            else (
                torch.tensor(lat_and_inv[0], dtype=_dtype),
                torch.tensor(lat_and_inv[1], dtype=_dtype),
            )
        )

        self.dim_d, self.n_train = model['R_desc'].shape[:2]
        self.dim_i = 3 * int((1 + np.sqrt(8 * self.dim_d + 1)) / 2)
        self.n_perms, self.n_atoms = model['perms'].shape

        # Check dublicates in permutation list.
        if model['perms'].shape[0] != np.unique(model['perms'], axis=0).shape[0]:
            self._log.warning('Model contains dublicate permutations')

        # Find index of identify permutation.
        self.idx_id_perm = np.where(
            (model['perms'] == np.arange(self.n_atoms)).all(axis=1)
        )[0]

        # No identity permutation found.
        if len(self.idx_id_perm) == 0:
            self._log.critical('Identity permutation is missing!')
            print()
            os._exit(1)

        # Identity permutation not at index zero.
        if len(self.idx_id_perm) > 0 and self.idx_id_perm[0] != 0:
            self._log.debug(
                'Identity is not at first position in permutation list (found at index {})'.format(
                    self.idx_id_perm[0]
                )
            )

        self.idx_id_perm = self.idx_id_perm[0]

        self._sig = int(model['sig'])
        self._c = float(model['c'])
        self._std = float(model.get('std', 1))

        self.tril_indices = np.tril_indices(self.n_atoms, k=-1)

        if _torch_cuda_is_available:  # Ignore limits and take whatever the GPU has.
            max_memory = (
                min(
                    [
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count())
                    ]
                )
                // 2**30
            )  # bytes to GB
        else:  # TODO: what about MPS?
            default_cpu_max_mem = 32
            if max_memory is None:
                self._log.warning(
                    'PyTorch CPU memory budget is limited to {} by default, which may impact performance.\n'.format(
                        ui.gen_memory_str(2**30 * default_cpu_max_mem)
                    )
                    + 'If necessary, adjust memory limit with option \'-m\'.'
                )
            max_memory = (
                max_memory or default_cpu_max_mem
            )  # 32 GB as default (hardcoded for now...)
        max_memory = int(2**30 * max_memory)  # GB to bytes

        min_const_mem, min_per_sample_mem = self.est_mem_requirement(return_min=True)

        log_type = (
            self._log.warning
            if min_const_mem + min_per_sample_mem >= max_memory
            else self._log.info
        )
        log_type(
            '{} memory report: max./avail. {}, min. req. (const./per-sample) ~{}/~{}'.format(
                'GPU'
                if (_torch_cuda_is_available or _torch_mps_is_available)
                else 'CPU',
                ui.gen_memory_str(max_memory),
                ui.gen_memory_str(min_const_mem),
                ui.gen_memory_str(min_per_sample_mem),
            )
        )

        self.max_processes = max_processes

        self.R_d_desc = None
        self._xs_train = nn.Parameter(
            torch.tensor(model['R_desc'], dtype=_dtype).t(), requires_grad=False
        )
        self._Jx_alphas = nn.Parameter(
            torch.tensor(np.array(model['R_d_desc_alpha']), dtype=_dtype),
            requires_grad=False,
        )

        self._alphas_E = None
        if 'alphas_E' in model:
            self._alphas_E = nn.Parameter(
                torch.from_numpy(model['alphas_E'], dtype=_dtype), requires_grad=False
            )

        self.perm_idxs = (
            torch.tensor(model['tril_perms_lin'], dtype=torch.long)
            .view(-1, self.n_perms)
            .t()
        )

        i, j = self.tril_indices
        self.register_buffer(
            'agg_mat', torch.zeros((self.n_atoms, self.dim_d), dtype=torch.int8)
        )
        self.agg_mat[i, range(self.dim_d)] = -1
        self.agg_mat[j, range(self.dim_d)] = 1

        # Try to cache all permutated variants of 'self._xs_train' and 'self._Jx_alphas'
        try:
            self.set_n_perm_batches(n_perm_batches)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if _torch_cuda_is_available:
                    torch.cuda.empty_cache()

                if n_perm_batches == 1:
                    self.set_n_perm_batches(
                        2
                    )  # Set to 2 perm batches, because that's the first batch size (and fastest) that is not cached.
                    pass
                else:
                    self._log.critical(
                        'Could not allocate enough memory to store model parameters on GPU. There is no hope!'
                    )
                    print()
                    os._exit(1)
            else:
                raise e

        const_mem, per_sample_mem = self.est_mem_requirement(return_min=False)
        _batch_size = (
            max((max_memory - const_mem) // per_sample_mem, 1)
            if batch_size is None
            else batch_size
        )
        max_batch_size = (
            self.n_train // torch.cuda.device_count()
            if _torch_cuda_is_available
            else self.n_train
        )
        _batch_size = min(_batch_size, max_batch_size)

        self._log.debug(
            'Setting batch size to {}/{} points.'.format(_batch_size, self.n_train)
        )

        self.desc = Desc(self.n_atoms, max_processes=max_processes)

    def get_n_perm_batches(self):

        global _n_perm_batches
        return _n_perm_batches

    def set_n_perm_batches(self, n_perm_batches):

        global _n_perm_batches

        self._log.debug(
            'Setting permutation batch size to {}/{}{}.'.format(
                self.n_perms // n_perm_batches,
                self.n_perms,
                ' (no caching)' if n_perm_batches > 1 else '',
            )
        )

        _n_perm_batches = n_perm_batches
        if n_perm_batches == 1 and self.n_perms > 1:
            self.cache_perms()
        else:
            self.uncache_perms()

    def apply_perms_to_obj(self, xs, perm_idxs=None):

        n_perms = 1 if perm_idxs is None else perm_idxs.numel() // self.dim_d
        perm_idxs = (
            slice(None) if perm_idxs is None else perm_idxs
        )  # slice(None) same as [:]

        # might run out of memory here, which will be handled by the caller
        try:
            return xs.repeat(1, n_perms)[:, perm_idxs].reshape(-1, self.dim_d)
        except:
            raise

    def remove_perms_from_obj(self, xs):

        return xs.reshape(self.n_train, -1, self.dim_d)[:, self.idx_id_perm, :].reshape(
            -1, self.dim_d
        )

    def uncache_perms(self):

        xs_train_n_perms = self._xs_train.numel() // (self.n_train * self.dim_d)
        if xs_train_n_perms != 1:  # Uncached already?
            self._xs_train = nn.Parameter(
                self.remove_perms_from_obj(self._xs_train), requires_grad=False
            )

        Jx_alphas_n_perms = self._Jx_alphas.numel() // (self.n_train * self.dim_d)
        if Jx_alphas_n_perms != 1:  # Uncached already?
            self._Jx_alphas = nn.Parameter(
                self.remove_perms_from_obj(self._Jx_alphas), requires_grad=False
            )

    def cache_perms(self):

        xs_train_n_perms = self._xs_train.numel() // (self.n_train * self.dim_d)
        if xs_train_n_perms == 1:  # Cached already?
            self._xs_train = nn.Parameter(
                self.apply_perms_to_obj(self._xs_train, perm_idxs=self.perm_idxs),
                requires_grad=False,
            )

        Jx_alphas_n_perms = self._Jx_alphas.numel() // (self.n_train * self.dim_d)
        if Jx_alphas_n_perms == 1:  # Cached already?
            self._Jx_alphas = nn.Parameter(
                self.apply_perms_to_obj(self._Jx_alphas, perm_idxs=self.perm_idxs),
                requires_grad=False,
            )

    def est_mem_requirement(self, return_min=False):
        """
        Calculate an estimate for the maximum/minimum memory needed to generate
        a prediction for a single geometry.

        Parameters
        ----------
        return_min : boolean, optional
            Return a minimum estimate instead.

        Returns
        -------
        const_mem : int
            Constant memory overhead (bytes) (allocated upon instantiation of the class)
        per_sample_mem : int
            Memory requirement for a single prediction (bytes)
        """

        n_perms_mem = 1 if return_min else self.n_perms

        # Constant memory requirement (bytes)
        const_mem = self.n_train * self.n_atoms * 3  # Rs (all)
        const_mem += n_perms_mem * self.dim_d  # perm_idxs
        const_mem += (
            n_perms_mem * self.n_train * self.dim_d * 2
        )  # _xs_train and _Jx_alphas
        const_mem += self.n_atoms * self.dim_d  # agg_mat
        const_mem *= 8
        const_mem = int(const_mem)

        # Peak memory requirement (bytes)
        per_sample_mem = 2 * self.n_atoms * 3  # Rs (batch), # Fs (batch)
        per_sample_mem += self.n_atoms  # Es (batch)
        per_sample_mem += self.n_atoms**2 * 3  # diffs
        per_sample_mem += self.dim_d  # xs
        per_sample_mem += self.dim_d * n_perms_mem * self.n_train  # x_diffs
        per_sample_mem += (
            4 * n_perms_mem * self.n_train
        )  # x_dists, exp_xs, dot_x_diff_Jx_alphas, exp_xs_1_x_dists
        per_sample_mem *= 8
        per_sample_mem = int(
            2 * per_sample_mem
        )  # HACK!!! Assume double that is needed. Seems to work better, maybe because of fragmentation issues?

        # <class 'torch.Tensor'> torch.Size([21, 118, 3]) # Fs
        # <class 'torch.Tensor'> torch.Size([21]) # Es
        # <class 'torch.Tensor'> torch.Size([21, 118, 3]) # Rs (batch)
        # <class 'torch.Tensor'> torch.Size([21, 118, 118, 3]) # diffs
        # <class 'torch.Tensor'> torch.Size([21, 6903]) # xs
        # <class 'torch.Tensor'> torch.Size([21, 5760, 6903])
        # <class 'torch.Tensor'> torch.Size([21, 5760]) # x_dists
        # <class 'torch.Tensor'> torch.Size([21, 5760]) # exp_xs
        # <class 'torch.Tensor'> torch.Size([21, 5760]) # dot_x_diff_Jx_alphas
        # <class 'torch.Tensor'> torch.Size([21, 5760]) # exp_xs_1_x_dists
        # <class 'torch.Tensor'> torch.Size([96, 6903]) # perm_idxs
        # <class 'torch.nn.parameter.Parameter'> torch.Size([5760, 6903]) # _xs_train
        # <class 'torch.nn.parameter.Parameter'> torch.Size([5760, 6903]) # _Jx_alphas
        # <class 'torch.Tensor'> torch.Size([60, 118, 3]) # Rs (all)

        return const_mem, per_sample_mem

    def set_R_d_desc(self, R_d_desc):
        """
        Set reference to training descriptor Jacobians. They are needed when the
        alpha coefficients are updated during iterative model training.

        This routine will try to move them to the GPU memory, if enough is available.

        Parameters
        ----------
        R_d_desc : :obj:`numpy.ndarray`
            Array containing the Jacobian of the descriptor for
            each training point.
        """

        self.R_d_desc = torch.from_numpy(R_d_desc).type(_dtype)

        # Try moving to GPU memory.
        if _torch_cuda_is_available or _torch_mps_is_available:
            try:
                R_d_desc = self.R_d_desc.to(self._xs_train.device)
            except RuntimeError as e:
                if 'out of memory' in str(e):

                    if _torch_cuda_is_available:
                        torch.cuda.empty_cache()

                    self._log.debug('Failed to cache \'R_d_desc\' on GPU.')
                else:
                    raise e
            else:
                self.R_d_desc = R_d_desc

    def set_alphas(self, alphas, alphas_E=None):
        """
        Reconfigure the current model with a new set of regression parameters.

        This routine is used during iterative model training.

        Parameters
        ----------
                alphas : :obj:`numpy.ndarray`
                    1D array containing the new model parameters.
                alphas_E : :obj:`numpy.ndarray`, optional
                    1D array containing the additional new model parameters, if
                    energy constraints are used in the kernel (`use_E_cstr=True`)
        """

        global _n_perm_batches

        if self.R_d_desc is None:
            self._log.critical(
                'The function \'set_alphas()\' requires \'R_d_desc\' to be set beforehand!'
            )
            print()
            os._exit(1)

        if alphas_E is not None:
            self._alphas_E = nn.Parameter(
                torch.from_numpy(alphas_E).to(self._xs_train.device).type(_dtype),
                requires_grad=False,
            )

        del self._Jx_alphas
        while True:
            try:

                alphas_torch = (
                    torch.from_numpy(alphas).type(_dtype).to(self.R_d_desc.device)
                )  # Send to whatever device 'R_d_desc' is on, first.
                xs = self.desc.d_desc_dot_vec(
                    self.R_d_desc, alphas_torch.reshape(-1, self.dim_i)
                )
                del alphas_torch

                if (_torch_cuda_is_available and not xs.is_cuda) or (
                    _torch_mps_is_available and not xs.is_mps
                ):
                    xs = xs.to(
                        self._xs_train.device
                    )  # Only now send it to the GPU ('_xs_train' will be for sure, if GPUs are available)

            except RuntimeError as e:
                if 'out of memory' in str(e):

                    if _torch_cuda_is_available or _torch_mps_is_available:

                        if _torch_cuda_is_available:
                            torch.cuda.empty_cache()

                        self.R_d_desc = self.R_d_desc.cpu()

                        self._log.debug(
                            'Failed to \'set_alphas()\': \'R_d_desc\' was moved back from GPU to CPU'
                        )

                        pass

                    else:

                        self._log.critical(
                            'Not enough memory to cache \'R_d_desc\'! There nothing we can do...'
                        )
                        print()
                        os._exit(1)

                else:
                    raise e
            else:
                break

        try:

            perm_idxs = self.perm_idxs if _n_perm_batches == 1 else None
            self._Jx_alphas = nn.Parameter(
                self.apply_perms_to_obj(xs, perm_idxs=perm_idxs), requires_grad=False
            )

        except RuntimeError as e:
            if 'out of memory' in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if _n_perm_batches < self.n_perms:

                    self._log.debug(
                        'Setting permutation batch size to {}/{}{}.'.format(
                            self.n_perms // n_perm_batches,
                            self.n_perms,
                            ' (no caching)' if n_perm_batches > 1 else '',
                        )
                    )

                    _n_perm_batches += 1  # Do NOT change me to use 'self.set_n_perm_batches(_n_perm_batches + 1)'!
                    self._xs_train = nn.Parameter(
                        self.remove_perms_from_obj(self._xs_train), requires_grad=False
                    )  # Remove any permutations from 'self._xs_train'.
                    self._Jx_alphas = nn.Parameter(
                        self.apply_perms_to_obj(xs, perm_idxs=None), requires_grad=False
                    )  # Set 'self._Jx_alphas' without applying permutations.

                else:
                    self._log.critical(
                        'Could not allocate enough memory to set new alphas in model.'
                    )
                    print()
                    os._exit(1)
            else:
                raise e

    def _forward(self, Rs_or_train_idxs, return_E=True):

        global _n_perm_batches

        q = np.sqrt(5) / self._sig
        i, j = self.tril_indices

        is_train_pred = Rs_or_train_idxs.dim() == 1
        if not is_train_pred:  # Rs

            Rs = Rs_or_train_idxs.type(_dtype)
            diffs = Rs[:, :, None, :] - Rs[:, None, :, :]  # N, a, a, 3
            diffs = diffs[:, i, j, :]  # N, d, 3

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

            xs = 1 / diffs.norm(dim=-1)  # N, d

            diffs *= xs[:, :, None] ** 3
            Jxs = diffs
            del diffs

        else:  # xs_train

            train_idxs = Rs_or_train_idxs

            # Get index of identity permutation, depending on caching configuration.
            xs_train_n_perms = self._xs_train.numel() // (self.n_train * self.dim_d)
            idx_id_perm = 0 if xs_train_n_perms == 1 else self.idx_id_perm

            xs = self._xs_train.reshape(self.n_train, -1, self.dim_d)[
                train_idxs, idx_id_perm, :
            ]  # ignore permutations

            Jxs = self.R_d_desc[train_idxs, :, :].to(
                xs.device
            )  # 'R_d_desc' might be living on the CPU...

        # current:
        # diffs: N, a, a, 3
        # xs: # N, d

        Fs_x = torch.zeros(xs.shape, device=xs.device, dtype=xs.dtype)
        Es = (
            torch.zeros((xs.shape[0],), device=xs.device, dtype=xs.dtype)
            if return_E
            else None
        )

        n_perms_done = 0
        for perm_batch in np.array_split(np.arange(self.n_perms), _n_perm_batches):

            if _n_perm_batches == 1:
                xs_train_perm_split = self._xs_train
                Jx_alphas_perm_split = self._Jx_alphas
            else:
                perm_idxs_batch = (
                    self.perm_idxs[perm_batch, :] - n_perms_done * self.dim_d
                )  # index shift
                xs_train_perm_split = self.apply_perms_to_obj(
                    self._xs_train, perm_idxs=perm_idxs_batch
                )
                Jx_alphas_perm_split = self.apply_perms_to_obj(
                    self._Jx_alphas, perm_idxs=perm_idxs_batch
                )

            n_perms_done += len(perm_batch)

            x_diffs = q * (
                xs[:, None, :] - xs_train_perm_split
            )  # N, n_perms*N_train, d
            x_dists = x_diffs.norm(dim=-1)  # N, n_perms*N

            exp_xs = torch.exp(-x_dists) * (q**2) / 3  # N, n_perms
            exp_xs_1_x_dists = exp_xs * (1 + x_dists)  # N, n_perms*N_train

            if self._alphas_E is None:
                del x_dists

            dot_x_diff_Jx_alphas = torch.einsum(
                'ij...,j...->ij', x_diffs, Jx_alphas_perm_split
            )  # N, n_perms*N_train

            # Fs_x = ((exp_xs * dot_x_diff_Jx_alphas)[..., None] * x_diffs).sum(dim=1)
            Fs_x += torch.einsum(  # NOTE ! Fs_x = Fs_x + torch.einsum(
                '...j,...j,...jk', exp_xs, dot_x_diff_Jx_alphas, x_diffs
            )  # N, d
            del exp_xs

            if self._alphas_E is None:
                del x_diffs

            # current:
            # diffs: N, a, a, 3
            # xs: # N, d
            # x_diffs: # N, n_perms*N_train, d
            # x_dists: # N, n_perms*N_train
            # exp_xs: # N, n_perms*N_train
            # dot_x_diff_Jx_alphas: N, n_perms*N_train
            # exp_xs_1_x_dists: N, n_perms*N_train
            # Fs_x: N, d

            Fs_x -= exp_xs_1_x_dists.mm(Jx_alphas_perm_split)  # N, d

            if return_E:
                Es += (
                    torch.einsum('...j,...j', exp_xs_1_x_dists, dot_x_diff_Jx_alphas)
                    / q
                )

            del dot_x_diff_Jx_alphas

            if self._alphas_E is None:
                del exp_xs_1_x_dists

            # Note: Energies are automatically predicted with a flipped sign here (because -E are trained, instead of E)
            if self._alphas_E is not None:

                K_fe = (x_diffs / q) * exp_xs_1_x_dists[:, :, None]
                del exp_xs_1_x_dists
                del x_diffs

                K_fe = K_fe.reshape(-1, self.n_train, len(perm_batch), self.dim_d)
                Fs_x += torch.einsum('j,...jkl->...l', self._alphas_E, K_fe)
                del K_fe

                K_ee = (1 + x_dists * (1 + x_dists / 3)) * torch.exp(-x_dists)
                del x_dists

                K_ee = K_ee.reshape(-1, self.n_train, len(perm_batch))
                Es += torch.einsum('j,...jk->...', self._alphas_E, K_ee)
                del K_ee

        # current:
        # diffs: N, a, a, 3
        # xs: # N, d
        # x_dists: # N, n_perms*N
        # dot_x_diff_Jx_alphas: N, n_perms*N
        # exp_xs_1_x_dists: N, n_perms*N
        # Fs_x: N, d

        Fs = torch.einsum('ji,...ik,...i->...jk', self.agg_mat.double(), Jxs, Fs_x)

        if not is_train_pred:  # TODO: set std to zero in training mode?
            Fs *= self._std

        if return_E:
            Es *= self._std
            Es += self._c

        return Es, Fs

    def forward(self, Rs_or_train_idxs, return_E=True):
        """
        Predict energy and forces for a batch of geometries.

        Parameters
        ----------
        Rs_or_train_idxs : :obj:`torch.Tensor`
            (dims M x N x 3) Cartesian coordinates of M molecules composed of N atoms or
            (dims N) index list of training points to evaluate. Note that `self.R_d_desc`
            needs to be set for the latter to work.
        return_E : boolean, optional
            If false (default: true), only the forces are returned.

        Returns
        -------
        E : :obj:`torch.Tensor`
            (dims M) Molecular energies (unless `return_E == False`)
        F : :obj:`torch.Tensor`
            (dims M x N x 3) Nuclear gradients of the energy
        """

        global _batch_size, _n_perm_batches

        # if Rs_or_train_idxs.dim() == 1:
        #    # contains index list. return predictions for these training points
        #    dtype = self.R_d_desc.dtype
        # elif Rs_or_train_idxs.dim() == 3:
        # this is real data

        #    assert Rs_or_train_idxs.shape[1:] == (self.n_atoms, 3)
        #    Rs_or_train_idxs = Rs_or_train_idxs.double()
        #    dtype = Rs_or_train_idxs.dtype

        # else:
        #    # unknown input
        #    self._log.critical('Invalid input for \'Rs_or_train_idxs\'.')
        #    print()
        #    os._exit(1)

        while True:
            try:
                Es, Fs = zip(
                    *map(
                        partial(self._forward, return_E=return_E),
                        DataLoader(Rs_or_train_idxs, batch_size=_batch_size),
                    )
                )
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if _batch_size > 1:

                        self._log.debug(
                            'Setting batch size to {}/{} points.'.format(
                                _batch_size, self.n_train
                            )
                        )
                        _batch_size -= 1

                    elif _n_perm_batches < self.n_perms:
                        n_perm_batches = _next_batch_size(self.n_perms, _n_perm_batches)
                        self.set_n_perm_batches(n_perm_batches)

                    else:
                        self._log.critical(
                            'Could not allocate enough (GPU) memory to evaluate model, despite reducing batch size.'
                        )
                        print()
                        os._exit(1)
                else:
                    raise e
            else:
                break

        ret = (torch.cat(Fs),)
        if return_E:
            ret = (torch.cat(Es),) + ret

        return ret

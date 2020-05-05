"""
This module contains all routines for training GDML and sGDML models.
"""

# MIT License
#
# Copyright (c) 2018-2020 Stefan Chmiela
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

from __future__ import print_function

import sys
import logging
import multiprocessing as mp
import inspect
import timeit
import warnings
from functools import partial
#from psutil import virtual_memory

import numpy as np
import scipy as sp

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True

from . import __version__
from .predict import GDMLPredict
from .utils.desc import Desc
from .utils import io, perm, ui


def _share_array(arr_np, typecode_or_type):
    """
    Return a ctypes array allocated from shared memory with data from a
    NumPy array.

    Parameters
    ----------
        arr_np : :obj:`numpy.ndarray`
            NumPy array.
        typecode_or_type : char or :obj:`ctype`
            Either a ctypes type or a one character typecode of the
            kind used by the Python array module.

    Returns
    -------
        array of :obj:`ctype`
    """

    arr = mp.RawArray(typecode_or_type, arr_np.ravel())
    return arr, arr_np.shape


def _assemble_kernel_mat_wkr(
    j, tril_perms_lin, sig, use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):
    r"""
    Compute one row and column of the force field kernel matrix.

    The Hessian of the Matern kernel is used with n = 2 (twice
    differentiable). Each row and column consists of matrix-valued
    blocks, which encode the interaction of one training point with all
    others. The result is stored in shared memory (a global variable).

    Parameters
    ----------
        j : int
            Index of training point.
        tril_perms_lin : :obj:`numpy.ndarray`
            1D array (int) containing all recovered permutations
            expanded as one large permutation to be applied to a tiled
            copy of the object to be permuted.
        sig : int
            Hyper-parameter :math:`\sigma`.
        use_E_cstr : bool, optional
            True: include energy constraints in the kernel,
            False: default (s)GDML kernel.
        exploit_sym : boolean, optional
            Do not create symmetric entries of the kernel matrix twice
            (this only works for spectific inputs for `cols_m_limit`)
        cols_m_limit : int, optional
            Limit the number of columns (include training points 1-`M`).
            Note that each training points consists of multiple columns.

    Returns
    -------
        int
            Number of kernel matrix blocks created, divided by 2
            (symmetric blocks are always created at together).
    """

    global glob

    R_desc = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
    R_d_desc = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])

    n_train, dim_d, dim_i = R_d_desc.shape
    n_perms = int(len(tril_perms_lin) / dim_d)

    if type(j) is tuple:  # selective/"fancy" indexing

        K_j, j, keep_idxs_3n = (
            j
        )  # (block index in final K, block index global, indices of partials within block)
        blk_j = slice(K_j, K_j + len(keep_idxs_3n))

    else:  # sequential indexing
        blk_j = slice(j * dim_i, (j + 1) * dim_i)
        keep_idxs_3n = slice(None)  # same as [:]

    # TODO: document this exception
    if use_E_cstr and not (cols_m_limit is None or cols_m_limit == n_train):
        raise ValueError(
            '\'use_E_cstr\'- and \'cols_m_limit\'-parameters are mutually exclusive!'
        )

    # Create permutated variants of 'rj_desc' and 'rj_d_desc'.
    rj_desc_perms = np.reshape(
        np.tile(R_desc[j, :], n_perms)[tril_perms_lin], (n_perms, -1), order='F'
    )
    rj_d_desc_perms = np.reshape(
        np.tile(R_d_desc[j, :, :].T, n_perms)[:, tril_perms_lin], (-1, dim_d, n_perms)
    )

    mat52_base_div = 3 * sig ** 4
    sqrt5 = np.sqrt(5.0)
    sig_pow2 = sig ** 2

    for i in range(j if exploit_sym else 0, n_train):
        blk_i = slice(i * dim_i, (i + 1) * dim_i)

        diff_ab_perms = R_desc[i, :] - rj_desc_perms
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

        mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5
        diff_ab_outer_perms = 5 * np.einsum(
            'ki,kj->ij',
            diff_ab_perms * mat52_base_perms[:, None],
            np.einsum('ik,jki -> ij', diff_ab_perms, rj_d_desc_perms),
        )
        diff_ab_outer_perms -= np.einsum(
            'ijk,k->ji',
            rj_d_desc_perms,
            (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
        )

        ## K[blk_i, blk_j] = K[blk_j, blk_i] = R_d_desc[i, :, :].T.dot(diff_ab_outer_perms)
        # K[blk_i, blk_j] = R_d_desc[i, :, :].T.dot(diff_ab_outer_perms)
        # if (
        #    i < cols_m_limit
        # ):  # symmetric extension is not always possible, if a partial kernel is assembled
        #    K[blk_j, blk_i] = K[blk_i, blk_j]R_d_desc :

        k = R_d_desc[i, :, :].T.dot(diff_ab_outer_perms)
        # K[blk_i, blk_j_keep[None, :]] = k[:, cols_3n_keep_idxs]
        # K[blk_i, blk_j[None, :]] = k[:, keep_idxs_3n]

        # if i < cols_m_limit:  # symmetric extension is not always possible, if a partial kernel is assembled
        #    K[blk_j[:, None], blk_i_keep.T] = k[cols_3n_keep_idxs, :].T

        K[blk_i, blk_j] = k[:, keep_idxs_3n]
        if exploit_sym and (
            cols_m_limit is None or i < cols_m_limit
        ):  # this will never be called with 'keep_idxs_3n' set to anything else than [:]
            K[blk_j, blk_i] = k.T

    if use_E_cstr:

        E_off = K.shape[0] - n_train, K.shape[1] - n_train
        blk_j_full = slice(j * dim_i, (j + 1) * dim_i)
        for i in range(n_train):

            diff_ab_perms = R_desc[i, :] - rj_desc_perms
            norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

            K_fe = (
                5
                * diff_ab_perms
                / (3 * sig ** 3)
                * (norm_ab_perms[:, None] + sig)
                * np.exp(-norm_ab_perms / sig)[:, None]
            )
            K_fe = -np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
            K[blk_j_full, E_off[1] + i] = K_fe  # vertical
            K[E_off[0] + i, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal

            # K[E_off[0] + i, blk_j] = K[blk_j_full, E_off[1] + i] = -np.einsum(
            #    'ik,jki -> j', K_fe, rj_d_desc_perms
            # )

            K[E_off[0] + i, E_off[1] + j] = K[E_off[0] + j, E_off[1] + i] = -(
                1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            ).dot(np.exp(-norm_ab_perms / sig))

    return blk_j.stop - blk_j.start  # TODO: fix me


class GDMLTrain(object):
    def __init__(self, max_processes=None, use_torch=False):
        """
        Train sGDML force fields.

        This class is used to train models using different closed-form
        and numerical solvers. GPU support is provided
        through PyTorch (requires optional `torch` dependency to be
        installed) for some solvers.

        Parameters
        ----------
                max_processes : int, optional
                        Limit the max. number of processes. Otherwise
                        all CPU cores are used. This parameters has no
                        effect if `use_torch=True`
                use_torch : boolean, optional
                        Use PyTorch to calculate predictions (if
                        supported by solver)

        Raises
        ------
            Exception
                If multiple instsances of this class are created.
            ImportError
                If the optional PyTorch dependency is missing, but PyTorch features are used.
        """

        global glob
        if 'glob' not in globals():  # Don't allow more than one instance of this class.
            glob = {}
        else:
            raise Exception(
                'You can not create multiple instances of this class. Please reuse your first one.'
            )

        self.log = logging.getLogger(__name__)

        self._max_processes = max_processes
        self._use_torch = use_torch

        if use_torch and not _has_torch:
            raise ImportError(
                'Optional PyTorch dependency not found! Please run \'pip install sgdml[torch]\' to install it or disable the PyTorch option.'
            )

    def __del__(self):

        global glob
        del glob

    def create_task(
        self,
        train_dataset,
        n_train,
        valid_dataset,
        n_valid,
        sig,
        lam=1e-15,
        use_sym=True,
        use_E=True,
        use_E_cstr=False,
        use_cprsn=False,
        model0=None,  # TODO: document me
        solver='analytic',  # TODO: document me
        solver_tol=1e-4,  # TODO: document me
        toggle_callback=None,  # TODO: document me
        progr_callback=None,  # TODO: document me
    ):
        """
        Create a data structure of custom type `task`.

        These data structures serve as recipes for model creation,
        summarizing the configuration of one particular training run.
        Training and test points are sampled from the provided dataset,
        without replacement. If the same dataset if given for training
        and testing, the subsets are drawn without overlap.

        Each task also contains a choice for the hyper-parameters of the
        training process and the MD5 fingerprints of the used datasets.

        Parameters
        ----------
            train_dataset : :obj:`dict`
                Data structure of custom type :obj:`dataset` containing
                train dataset.
            n_train : int
                Number of training points to sample.
            valid_dataset : :obj:`dict`
                Data structure of custom type :obj:`dataset` containing
                validation dataset.
            n_valid : int
                Number of validation points to sample.
            sig : int
                Hyper-parameter (kernel length scale).
            lam : float, optional
                Hyper-parameter lambda (regularization strength).
            use_sym : bool, optional
                True: include symmetries (sGDML), False: GDML.
            use_E : bool, optional
                True: reconstruct force field with corresponding potential energy surface,
                False: ignore energy during training, even if energy labels are available
                       in the dataset. The trained model will still be able to predict
                       energies up to an unknown integration constant. Note, that the
                       energy predictions accuracy will be untested.
            use_E_cstr : bool, optional
                True: include energy constraints in the kernel,
                False: default (s)GDML.
            use_cprsn : bool, optional
                True: compress kernel matrix along symmetric degrees of
                freedom,
                False: train using full kernel matrix
            model0 : :obj:`dict`, optional
                Create a task based on an existing template model. Training and validation
                splits are reused, so are the permutations and regression parameters.

        Returns
        -------
            dict
                Data structure of custom type :obj:`task`.

        Raises
        ------
            ValueError
                If a reconstruction of the potential energy surface is requested,
                but the energy labels are missing in the dataset.
        """

        if use_E and 'E' not in train_dataset:
            raise ValueError(
                'No energy labels found in dataset!\n'
                + 'By default, force fields are always reconstructed including the\n'
                + 'corresponding potential energy surface (this can be turned off).\n'
                + 'However, the energy labels are missing in the provided dataset.\n'
            )

        use_E_cstr = use_E and use_E_cstr

        n_atoms = train_dataset['R'].shape[1]

        if toggle_callback is not None:
            toggle_callback = partial(toggle_callback, disp_str='Hashing dataset(s)')
            toggle_callback(is_done=False)

        md5_train = io.dataset_md5(train_dataset)
        md5_valid = io.dataset_md5(valid_dataset)

        if toggle_callback is not None:
            toggle_callback(is_done=True)

        if model0 is not None and (
            md5_train != model0['md5_train'] or md5_valid != model0['md5_valid']
        ):
            raise ValueError(
                'Provided training and/or validation dataset(s) do(es) not match the ones in the initial model.'
            )

        m0_excl_idxs = np.array([], dtype=np.uint)
        m0_n_train, m0_n_valid = 0, 0
        if model0 is not None:

            m0_idxs_train = model0['idxs_train']
            m0_idxs_valid = model0['idxs_valid']

            m0_n_train = m0_idxs_train.shape[0]
            m0_n_valid = m0_idxs_valid.shape[0]

            m0_excl_idxs = np.concatenate((m0_idxs_train, m0_idxs_valid)).astype(
                np.uint
            )

        # TODO: handle smaller training/validation set

        if toggle_callback is not None:
            toggle_callback = partial(
                toggle_callback, disp_str='Sampling training and validation subsets'
            )
            toggle_callback(is_done=False)

        if 'E' in train_dataset:
            idxs_train = self.draw_strat_sample(
                train_dataset['E'], n_train - m0_n_train, m0_excl_idxs
            )
        else:
            idxs_train = np.random.choice(
                np.arange(train_dataset['F'].shape[0]),
                n_train - m0_n_train,
                replace=False,
            )
            # TODO: m0 handling

        excl_idxs = (
            idxs_train if md5_train == md5_valid else np.array([], dtype=np.uint)
        )  # TODO: TEST CASE: differnt test and val sets and m0
        excl_idxs = np.concatenate((m0_excl_idxs, excl_idxs)).astype(np.uint)

        if 'E' in valid_dataset:
            idxs_valid = self.draw_strat_sample(
                valid_dataset['E'], n_valid - m0_n_valid, excl_idxs
            )
        else:
            idxs_valid_all = np.setdiff1d(
                np.arange(valid_dataset['F'].shape[0]), excl_idxs, assume_unique=True
            )
            idxs_valid = np.random.choice(
                idxs_valid_all, n_valid - m0_n_valid, replace=False
            )
            # TODO: m0 handling, zero handling

        if toggle_callback is not None:
            toggle_callback(is_done=True)

        if model0 is not None:
            idxs_train = np.concatenate((m0_idxs_train, idxs_train)).astype(np.uint)
            idxs_valid = np.concatenate((m0_idxs_valid, idxs_valid)).astype(np.uint)

        R_train = train_dataset['R'][idxs_train, :, :]
        task = {
            'type': 't',
            'code_version': __version__,
            'dataset_name': train_dataset['name'].astype(str),
            'dataset_theory': train_dataset['theory'].astype(str),
            'z': train_dataset['z'],
            'R_train': R_train,
            'F_train': train_dataset['F'][idxs_train, :, :],
            'idxs_train': idxs_train,
            'md5_train': md5_train,
            'idxs_valid': idxs_valid,
            'md5_valid': md5_valid,
            'sig': sig,
            'lam': lam,
            'use_E': use_E,
            'use_E_cstr': use_E_cstr,
            'use_sym': use_sym,
            'use_cprsn': use_cprsn,
            'solver_name': solver,
            'solver_tol': solver_tol,
        }

        if use_E:
            task['E_train'] = train_dataset['E'][idxs_train]

        lat_and_inv = None
        if 'lattice' in train_dataset:
            task['lattice'] = train_dataset['lattice']

            
            #if 'lattice' in train_dataset:
            try:
                lat_and_inv = (task['lattice'], np.linalg.inv(task['lattice']))
            except np.linalg.LinAlgError:
                raise ValueError(  # TODO: Document me
                    'Provided dataset contains invalid lattice vectors (not invertible). Note: Only rank 3 lattice vector matrices are supported.'
                )

        if 'r_unit' in train_dataset and 'e_unit' in train_dataset:
            task['r_unit'] = train_dataset['r_unit']
            task['e_unit'] = train_dataset['e_unit']

        if model0 is None:
            if use_sym:
                n_train = R_train.shape[0]
                R_train_sync_mat = R_train
                if n_train > 1000:
                    R_train_sync_mat = R_train[
                        np.random.choice(n_train, 1000, replace=False), :, :
                    ]
                    self.log.info(
                        'Symmetry search has been restricted to a random subset of 1000/{:d} training points for faster convergence.'.format(
                            n_train
                        )
                    )

                task['perms'] = perm.find_perms(
                    R_train_sync_mat, train_dataset['z'], lat_and_inv=lat_and_inv, max_processes=self._max_processes,
                )
            else:
                task['perms'] = np.arange(train_dataset['R'].shape[1])[
                    None, :
                ]  # no symmetries
        else:
            task['perms'] = model0['perms']
            self.log.info('Reusing permutations from initial model.')

        if model0 is not None:

            n_train, n_atoms = task['R_train'].shape[:2]

            if 'alphas_F' in model0:
                self.log.info('Reusing alphas from initial model.')

                # Pad existing alphas, if this training dataset is larger than the one in model0
                alphas0_F_padding = np.ones(
                    ((n_train - m0_n_train) * n_atoms * 3,)
                ) * np.mean(model0['alphas_F'])
                task['alphas0_F'] = np.append(model0['alphas_F'], alphas0_F_padding)

            if 'alphas_E' in model0:

                # Pad existing alphas, if this training dataset is larger than the one in model0
                alphas0_E_padding = np.ones(
                    ((n_train - m0_n_train) * n_atoms,)
                ) * np.mean(model0['alphas_E'])
                task['alphas0_E'] = np.append(model0['alphas_E'], alphas0_E_padding)

        # Which atoms can we keep, if we exclude all symmetric ones?
        n_perms = task['perms'].shape[0]
        if use_cprsn and n_perms > 1:

            _, cprsn_keep_idxs = np.unique(
                np.sort(task['perms'], axis=0), axis=1, return_index=True
            )

            task['cprsn_keep_atoms_idxs'] = cprsn_keep_idxs

        # Select inducing columns for Nystrom preconditioner used in combination with conjugated gradients based on leverage score approximations.
        if (
            solver == 'cg'
        ):  # TODO: resuse indices, if same number of training poitns is used

            desc = Desc(
                n_atoms, max_processes=self._max_processes,
            )

            dim_d = desc.dim

            n_perms = task['perms'].shape[0]
            tril_perms = np.array([desc.perm(p) for p in task['perms']])

            perm_offsets = np.arange(n_perms)[:, None] * dim_d
            tril_perms_lin = (tril_perms + perm_offsets).flatten('F')

            # lat_and_inv = None
            # if 'lattice' in train_dataset:
            #     try:
            #         lat_and_inv = (task['lattice'], np.linalg.inv(task['lattice']))
            #     except np.linalg.LinAlgError:
            #         raise ValueError(  # TODO: Document me
            #             'Provided dataset contains invalid lattice vectors (not invertible). Note: Only rank 3 lattice vector matrices are supported.'
            #         )

            R_desc, R_d_desc = desc.from_R(
                R_train.reshape(n_train, -1), lat_and_inv=lat_and_inv
            )

            lev_approx_idxs, max_lev_idxs = self._lev_scores(
                R_desc,
                R_d_desc,
                tril_perms_lin,
                sig,
                lam,
                False,
                progr_callback=progr_callback,
            )

            task['lev_approx_idxs'] = lev_approx_idxs
            task['nystrom_col_idxs'] = max_lev_idxs

        # # consitency test

        # #print(np.linalg.norm(task['F_train'], axis=2).shape)

        # #print(task['perms'].shape)

        # F_train_mags = np.linalg.norm(task['F_train'], axis=2)

        # #acc = np.zeros(task['R_train'].shape)

        # #min_i = -1
        # #min_d = -1

        # ds = []

        # for i,r in enumerate(task['R_train']):

        #     pdist = sp.spatial.distance.pdist(r, 'euclidean')
        #     pdist = sp.spatial.distance.squareform(pdist, checks=False)

        #     acc = 0
        #     for p in task['perms']:

        #         #pdist_p = sp.spatial.distance.pdist(r[p,:], 'euclidean')

        #         pdist_p = pdist[p,:][:,p]

        #         acc += np.sum(np.abs(pdist-pdist_p))

        #     #d = np.sum(acc)
        #     ds.append(acc)

        #     #if min_d > d or min_d == -1:
        #     #    min_d = d
        #     #    min_i = i

        #         #print(np.mean(np.abs(F_train_mags[:,p]-F_train_mags)))
        #         #acc += np.abs(task['R_train'][:,p,:] - task['R_train'])

        # idx_sorted = np.argsort(ds)

        # fs = []
        # for i,f in enumerate(task['F_train']):
        #     f_norm = np.linalg.norm(f, axis=1)

        #     if i == 0:
        #         print(f_norm)

        #     acc = 0
        #     for p in task['perms']:
        #         acc += np.sum(np.abs(f_norm-f_norm[p]))

        #         if i == 0:
        #             print(f_norm[p])
        #             print(f_norm-f_norm[p])

        #     fs.append(acc)

        # ds = np.array(ds)
        # ds = ds / np.linalg.norm(ds)

        # fs = np.array(fs)
        # fs = fs / np.linalg.norm(fs)

        # #import matplotlib.pyplot as plt
        # #plt.plot(ds[idx_sorted])
        # #plt.plot(fs[idx_sorted])
        # #plt.show()

        # min_i = idx_sorted[0]
        # xyz_str = io.generate_xyz_str(task['R_train'][min_i], task['z'], f=task['F_train'][min_i])
        # print(xyz_str)

        # min_i = idx_sorted[-1]
        # xyz_str = io.generate_xyz_str(task['R_train'][min_i], task['z'], f=task['F_train'][min_i])
        # print(xyz_str)

        # sys.exit()

        return task

    def create_model(
        self,
        task,
        solver,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        std,
        alphas_F,
        alphas_E=None,
        solver_resid=None,
        solver_iters=None,
        lev_approx_idxs=None,  # NEW : which columns of the full kernel matrix were used to approximate the leverage scores?
        nystrom_col_idxs=None,  # NEW : which columns were used to construct nystrom preconditioner
    ):

        r_dim = R_d_desc.shape[2]
        r_d_desc_alpha = np.einsum('kji,ki->kj', R_d_desc, alphas_F.reshape(-1, r_dim))

        model = {
            'type': 'm',
            'code_version': __version__,
            'dataset_name': task['dataset_name'],
            'dataset_theory': task['dataset_theory'],
            'solver_name': solver,
            'z': task['z'],
            'idxs_train': task['idxs_train'],
            'md5_train': task['md5_train'],
            'idxs_valid': task['idxs_valid'],
            'md5_valid': task['md5_valid'],
            'n_test': 0,
            'md5_test': None,
            'f_err': {'mae': np.nan, 'rmse': np.nan},
            'R_desc': R_desc.T,
            'R_d_desc_alpha': r_d_desc_alpha,
            'c': 0.0,
            'std': std,
            'sig': task['sig'],
            'lam': task['lam'],
            'alphas_F': alphas_F,  # needed?
            'perms': task['perms'],
            'tril_perms_lin': tril_perms_lin,
            'use_E': task['use_E'],
            'use_cprsn': task['use_cprsn'],
        }

        if solver_resid is not None:
            model['solver_resid'] = solver_resid  # residual of solution (cg solver)

        if solver_iters is not None:
            model[
                'solver_iters'
            ] = (
                solver_iters
            )  # number of iterations performed to obtain solution (cg solver)

        if lev_approx_idxs is not None:
            model['lev_approx_idxs'] = lev_approx_idxs

        if nystrom_col_idxs is not None:
            model['nystrom_col_idxs'] = nystrom_col_idxs

        if task['use_E']:
            model['e_err'] = {'mae': np.nan, 'rmse': np.nan}

            if task['use_E_cstr']:
                model['alphas_E'] = alphas_E

        if 'lattice' in task:
            model['lattice'] = task['lattice']

        if 'r_unit' in task and 'e_unit' in task:
            model['r_unit'] = task['r_unit']
            model['e_unit'] = task['e_unit']

        return model

    def train(  # noqa: C901
        self,
        task,
        cprsn_callback=None,
        desc_callback=None,
        ker_progr_callback=None,
        solve_callback=None,
        save_progr_callback=None,  # TODO: document me
    ):
        """
        Train a model based on a training task.

        Parameters
        ----------
            task : :obj:`dict`
                Data structure of custom type :obj:`task`.
            cprsn_callback : callable, optional
                Symmetry compression status.
                    n_atoms : int
                        Total number of atoms.
                    n_atoms_kept : float or None, optional
                        Number of atoms kept after compression.
            desc_callback : callable, optional
                Descriptor and descriptor Jacobian generation status.
                    current : int
                        Current progress (number of completed descriptors).
                    total : int
                        Task size (total number of descriptors to create).
                    done_str : :obj:`str`, optional
                        Once complete, this string contains the
                        time it took complete this task (seconds).
            ker_progr_callback : callable, optional
                Kernel assembly progress function that takes three
                arguments:
                    current : int
                        Current progress (number of completed entries).
                    total : int
                        Task size (total number of entries to create).
                    done_str : :obj:`str`, optional
                        Once complete, this string contains the
                        time it took to assemble the kernel (seconds).
            solve_callback : callable, optional
                Linear system solver status.
                    done : bool
                        False when solver starts, True when it finishes.
                    done_str : :obj:`str`, optional
                        Once done, this string contains the runtime
                        of the solver (seconds).

        Returns
        -------
            :obj:`dict`
                Data structure of custom type :obj:`model`.

        Raises
        ------
            ValueError
                If the provided dataset contains invalid lattice
                vectors.
        """

        task = dict(task)  # make mutable

        solver = task['solver_name']
        assert solver == 'analytic' or solver == 'cg'  # or solver == 'fk'

        n_train, n_atoms = task['R_train'].shape[:2]

        desc = Desc(
            n_atoms, max_processes=self._max_processes
        )

        sig = np.squeeze(task['sig'])
        lam = np.squeeze(task['lam'])

        n_perms = task['perms'].shape[0]
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        dim_i = 3 * n_atoms
        dim_d = desc.dim

        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')

        # TODO: check if all atoms are in span of lattice vectors, otherwise suggest that
        # rows and columns might have been switched.
        lat_and_inv = None
        if 'lattice' in task:
            try:
                lat_and_inv = (task['lattice'], np.linalg.inv(task['lattice']))
            except np.linalg.LinAlgError:
                raise ValueError(  # TODO: Document me
                    'Provided dataset contains invalid lattice vectors (not invertible). Note: Only rank 3 lattice vector matrices are supported.'
                )

            # # TODO: check if all atoms are within unit cell
            # for r in task['R_train']:
            #     r_lat = lat_and_inv[1].dot(r.T)
            #     if not (r_lat >= 0).all():
            #         # raise ValueError( # TODO: Document me
            #         #    'Some atoms appear outside of the unit cell! Please check lattice vectors in dataset file.'
            #         # )
            #         pass

        R_desc, R_d_desc = desc.from_R(
            task['R_train'].reshape(n_train, -1),
            lat_and_inv=lat_and_inv,
            callback=desc_callback,
        )

        # Which columns to include in K?
        col_idxs = np.s_[:]  # all of them

        # Generate indices of kernel matrix columns that are kept after compression.
        cprsn_keep_idxs_lin = None
        if 'cprsn_keep_atoms_idxs' in task:

            cprsn_keep_idxs = task['cprsn_keep_atoms_idxs']
            cprsn_keep_idxs_lin = (
                np.arange(dim_i).reshape(n_atoms, -1)[cprsn_keep_idxs, :].ravel()
            )

            if cprsn_callback is not None:
                cprsn_callback(n_atoms, cprsn_keep_idxs.shape[0])

            if solver != 'analytic':
                raise ValueError(
                    'Iterative solvers and compression are mutually exclusive options for now.'
                )

            col_idxs = (
                cprsn_keep_idxs_lin[:, None] + np.arange(n_train) * dim_i
            ).T.ravel()

        # test

        # n = 0.1

        # test

        # Determine inducing points for Nystrom preconditioner.
        if solver == 'cg':

            if 'nystrom_col_idxs' in task:  # use inducing points from task

                lev_approx_idxs = task['lev_approx_idxs']
                max_lev_idxs = task['nystrom_col_idxs']

            else:  # determine good inducing points

                lev_approx_idxs, max_lev_idxs = self._lev_scores(
                    R_desc,
                    R_d_desc,
                    tril_perms_lin,
                    sig,
                    lam,
                    False,
                    progr_callback=ker_progr_callback,
                )

            col_idxs = max_lev_idxs
            ker_progr_callback = partial(
                ker_progr_callback,
                disp_str='Assembling partial kernel matrix (for Nystrom preconditioner)',
            )

        # Generate label vector.
        E_train_mean = None
        y = task['F_train'].ravel()
        if task['use_E'] and task['use_E_cstr']:
            E_train = task['E_train'].ravel()
            E_train_mean = np.mean(E_train)

            y = np.hstack((y, -E_train + E_train_mean))
            # y = np.hstack((n*Ft, (1-n)*Et))
        y_std = np.std(y)
        y /= y_std

        # Generate kernel matrix.
        K = self._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            use_E_cstr=task['use_E_cstr'],
            progr_callback=ker_progr_callback,
            col_idxs=col_idxs,
        )

        # test

        # rows
        # K[:(3*n_atoms*n_train),:] *= n # force
        # K[(3*n_atoms*n_train):,:] *= 1-n # energy

        # print(K[:(3*n_atoms*n_train),:].shape)
        # print(K[(3*n_atoms*n_train):,:].shape)

        # columns
        # K[:,:(3*n_atoms*n_train)] *= n # force
        # K[:,(3*n_atoms*n_train):] *= 1-n # energy

        # K[:(3*n_atoms*n_train),:(3*n_atoms*n_train)] *= 1 # force
        # K[-n_train:,-n_train:] *= 2-2*n # energy
        # K[:(3*n_atoms*n_train),-n_train:] *= n-1  # force energy contrib
        # K[-n_train:,:(3*n_atoms*n_train)] *= n-1  # energy force contrib

        # K[:(3*n_atoms*n_train),:(3*n_atoms*n_train)] *= n**2 # force
        # K[-n_train:,-n_train:] *= (1-n)**2 # energy
        # K[:(3*n_atoms*n_train),-n_train:] *= n*(1-n)  # force energy contrib
        # K[-n_train:,:(3*n_atoms*n_train)] *= n*(1-n)  # energy force contrib

        # test

        num_iters = None  # number of iterations performed (cg solver)
        res = None  # residual of solution (cg solver)

        if solver == 'analytic':

            if cprsn_keep_idxs_lin is not None:
                R_d_desc = R_d_desc[:, :, cprsn_keep_idxs_lin]

            alphas = self._solve_closed(K, y, lam, callback=solve_callback)

        elif solver == 'cg':

            alphas_F = task['alphas0_F'] if 'alphas0_F' in task else None
            alphas_E = task['alphas0_E'] if 'alphas0_E' in task else None

            # this solver needs stronger regularization than the default 1e-15
            task['lam'] = 1e-10

            alphas, num_iters, res = self._solve_iterative_nystrom_precon(
                K,
                y,
                R_desc,
                R_d_desc,
                task,
                tril_perms_lin,
                y_std,
                alphas0_F=alphas_F,
                alphas0_E=alphas_E,
                callback=solve_callback,
                save_progr_callback=save_progr_callback,
                lev_approx_idxs=lev_approx_idxs,
                nystrom_col_idxs=col_idxs,
            )

        else:
            raise ValueError(
                'Unknown solver keyword \'{}\'.'.format(solver)
            )  # TODO: refine

        alphas_E = None
        alphas_F = alphas
        if task['use_E_cstr']:
            alphas_E = alphas[-n_train:]
            alphas_F = alphas[:-n_train]

        model = self.create_model(
            task,
            solver,
            R_desc,
            R_d_desc,
            tril_perms_lin,
            y_std,
            alphas_F,
            alphas_E=alphas_E,
            solver_resid=res,
            solver_iters=num_iters,
            lev_approx_idxs=lev_approx_idxs if solver == 'cg' else None,
            nystrom_col_idxs=col_idxs if solver == 'cg' else None,
        )

        # Recover integration constant.
        # Note: if energy constraints are included in the kernel (via 'use_E_cstr'), do not
        # compute the integration constant, but simply set it to the mean of the training energies
        # (which was subtracted from the labels before training).
        if model['use_E']:
            c = (
                self._recov_int_const(model, task)
                if E_train_mean is None
                else E_train_mean
            )
            if c is None:
                # Something does not seem right. Turn off energy predictions for this model, only output force predictions.
                model['use_E'] = False
            else:
                model['c'] = c

        return model

    def _recov_int_const(self, model, task):  # TODO: document e_err_inconsist return
        """
        Estimate the integration constant for a force field model.

        The offset between the energies predicted for the original training
        data and the true energy labels is computed in the least square sense.
        Furthermore, common issues with the user-provided datasets are self
        diagnosed here.

        Parameters
        ----------
            model : :obj:`dict`
                Data structure of custom type :obj:`model`.
            task : :obj:`dict`
                Data structure of custom type :obj:`task`.

        Returns
        -------
            float
                Estimate for the integration constant.

        Raises
        ------
            ValueError
                If the sign of the force labels in the dataset from
                which the model emerged is switched (e.g. gradients
                instead of forces).
            ValueError
                If inconsistent/corrupted energy labels are detected
                in the provided dataset.
            ValueError
                If different scales in energy vs. force labels are
                detected in the provided dataset.
        """

        gdml = GDMLPredict(
            model, max_processes=self._max_processes
        )  # , use_torch=self._use_torch

        n_train = task['E_train'].shape[0]
        R = task['R_train'].reshape(n_train, -1)

        E_pred, _ = gdml.predict(R)
        E_ref = np.squeeze(task['E_train'])

        e_fact = np.linalg.lstsq(
            np.column_stack((E_pred, np.ones(E_ref.shape))), E_ref, rcond=-1
        )[0][0]
        corrcoef = np.corrcoef(E_ref, E_pred)[0, 1]

        # import matplotlib.pyplot as plt
        # plt.plot(E_ref-np.mean(E_ref))
        # plt.plot(np.sort(E_pred-np.mean(E_pred)))
        # plt.show()

        if np.sign(e_fact) == -1:
            self.log.warning(
                'The provided dataset contains gradients instead of force labels (flipped sign). Please correct!\n'
                + ui.color_str('Note:', bold=True)
                + 'Note: The energy prediction accuracy of the model will thus neither be validated nor tested in the following steps!'
            )
            return None

        if corrcoef < 0.95:
            self.log.warning(
                'Inconsistent energy labels detected!\n'
                + 'The predicted energies for the training data are only weakly correlated with the reference labels (correlation coefficient {:.2f}) which indicates that the issue is most likely NOT just a unit conversion error.\n\n'.format(
                    corrcoef
                )
                + ui.color_str('Troubleshooting tips:\n', bold=True)
                + ui.wrap_indent_str(
                    '(1) ',
                    'Verify the correct correspondence between geometries and labels in the provided dataset.',
                )
                + '\n'
                + ui.wrap_indent_str(
                    '(2) ', 'Verify the consistency between energy and force labels.'
                )
                + '\n'
                + ui.wrap_indent_str('    - ', 'Correspondence correct?')
                + '\n'
                + ui.wrap_indent_str('    - ', 'Same level of theory?')
                + '\n'
                + ui.wrap_indent_str('    - ', 'Accuracy of forces (if numerical)?')
                + '\n'
                + ui.wrap_indent_str(
                    '(3) ',
                    'Is the training data spread too broadly (i.e. weakly sampled transitions between example clusters)?',
                )
                + '\n'
                + ui.wrap_indent_str(
                    '(4) ', 'Are there duplicate geometries in the training data?'
                )
                + '\n'
                + ui.wrap_indent_str(
                    '(5) ', 'Are there any corrupted data points (e.g. parsing errors)?'
                )
                + '\n\n'
                + ui.color_str('Note:', bold=True)
                + ' The energy prediction accuracy of the model will thus neither be validated nor tested in the following steps!'
            )
            return None

        if np.abs(e_fact - 1) > 1e-1:
            self.log.warning(
                'Different scales in energy vs. force labels detected!\n'
                + 'The integrated forces differ from the energy labels by factor ~{:.2f}, meaning that the trained model will likely fail to predict energies accurately.\n\n'.format(
                    e_fact
                )
                + ui.color_str('Troubleshooting tips:\n', bold=True)
                + ui.wrap_indent_str(
                    '(1) ', 'Verify consistency of units in energy and force labels.'
                )
                + '\n'
                + ui.wrap_indent_str(
                    '(2) ',
                    'Is the training data spread too broadly (i.e. weakly sampled transitions between example clusters)?',
                )
                + '\n\n'
                + ui.color_str('Note:', bold=True)
                + ' The energy prediction accuracy of the model will thus neither be validated nor tested in the following steps!'
            )
            return None

        # Least squares estimate for integration constant.
        return np.sum(E_ref - E_pred) / E_ref.shape[0]

    def _assemble_kernel_mat(
        self,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        sig,
        use_E_cstr=False,
        progr_callback=None,
        col_idxs=np.s_[:],  # TODO: document me
    ):
        r"""
        Compute force field kernel matrix.

        The Hessian of the Matern kernel is used with n = 2 (twice
        differentiable). Each row and column consists of matrix-valued blocks,
        which encode the interaction of one training point with all others. The
        result is stored in shared memory (a global variable).

        Parameters
        ----------
            R_desc : :obj:`numpy.ndarray`
                Array containing the descriptor for each training point.
            R_d_desc : :obj:`numpy.ndarray`
                Array containing the gradient of the descriptor for
                each training point.
            tril_perms_lin : :obj:`numpy.ndarray`
                1D array containing all recovered permutations
                expanded as one large permutation to be applied to a
                tiled copy of the object to be permuted.
            sig : int
                Hyper-parameter :math:`\sigma`(kernel length scale).
            use_E_cstr : bool, optional
                True: include energy constraints in the kernel,
                False: default (s)GDML kernel.
            progress_callback : callable, optional
                Kernel assembly progress function that takes three
                arguments:
                    current : int
                        Current progress (number of completed entries).
                    total : int
                        Task size (total number of entries to create).
                    done_str : :obj:`str`, optional
                        Once complete, this string contains the
                        time it took to assemble the kernel (seconds).
            cols_m_limit : int, optional
                Only generate the columns up to index 'cols_m_limit'. This creates
                a M*3N x cols_m_limit*3N kernel matrix, instead of M*3N x M*3N.
            cols_3n_keep_idxs : :obj:`numpy.ndarray`, optional
                Only generate columns with the given indices in the 3N x 3N
                kernel function. The resulting kernel matrix will have dimension
                M*3N x M*len(cols_3n_keep_idxs).


        Returns
        -------
            :obj:`numpy.ndarray`
                Force field kernel matrix.
        """

        global glob

        # Note: This function does not support unsorted (ascending) index arrays.
        # if not isinstance(col_idxs, slice):
        #    assert np.array_equal(col_idxs, np.sort(col_idxs))

        n_train, dim_d, dim_i = R_d_desc.shape

        # Determine size of kernel matrix.
        K_n_rows = n_train * dim_i
        if isinstance(col_idxs, slice):  # indexed by slice
            K_n_cols = len(range(*col_idxs.indices(K_n_rows)))
        else:  # indexed by list

            # TODO: throw exeption with description
            assert len(col_idxs) == len(set(col_idxs))  # assume no dublicate indices

            # TODO: throw exeption with description
            # Note: This function does not support unsorted (ascending) index arrays.
            assert np.array_equal(col_idxs, np.sort(col_idxs))

            K_n_cols = len(col_idxs)

        # Account for additional rows and columns due to energy constraints in the kernel matrix.
        if use_E_cstr:
            K_n_rows += n_train
            K_n_cols += n_train

        # Make sure no indices are outside of the valid range.
        if K_n_cols > K_n_rows:
            raise ValueError('Columns indexed beyond range.')

        exploit_sym = False
        cols_m_limit = None

        # Check if range is a subset of training points (as opposed to a subset of partials of multiple points).
        is_M_subset = (
            isinstance(col_idxs, slice)
            and (col_idxs.start is None or col_idxs.start % dim_i == 0)
            and (col_idxs.stop is None or col_idxs.stop % dim_i == 0)
            and col_idxs.step is None
        )
        if is_M_subset:
            M_slice_start = (
                None if col_idxs.start is None else int(col_idxs.start / dim_i)
            )
            M_slice_stop = None if col_idxs.stop is None else int(col_idxs.stop / dim_i)
            M_slice = slice(M_slice_start, M_slice_stop)

            J = range(*M_slice.indices(n_train))

            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop

        else:

            if isinstance(col_idxs, slice):
                random = list(range(*col_idxs.indices(n_train * dim_i)))
            else:
                random = col_idxs

            # M - number training
            # N - number atoms

            n_idxs = np.mod(random, dim_i)
            m_idxs = (np.array(random) / dim_i).astype(int)
            m_idxs_uniq = np.unique(m_idxs)  # which points to include?

            m_n_idxs = [
                list(n_idxs[np.where(m_idxs == m_idx)]) for m_idx in m_idxs_uniq
            ]

            m_n_idxs_lens = [len(m_n_idx) for m_n_idx in m_n_idxs]

            m_n_idxs_lens.insert(0, 0)
            blk_start_idxs = list(
                np.cumsum(m_n_idxs_lens[:-1])
            )  # index within K at which each block starts

            # tupels: (block index in final K, block index global, indices of partials within block)
            J = list(zip(blk_start_idxs, m_idxs_uniq, m_n_idxs))

        K = mp.RawArray('d', K_n_rows * K_n_cols)
        glob['K'], glob['K_shape'] = K, (K_n_rows, K_n_cols)
        glob['R_desc'], glob['R_desc_shape'] = _share_array(R_desc, 'd')
        glob['R_d_desc'], glob['R_d_desc_shape'] = _share_array(R_d_desc, 'd')

        start = timeit.default_timer()
        pool = mp.Pool(self._max_processes)

        # todo = (cols_m_limit ** 2 - cols_m_limit) // 2 + cols_m_limit
        # if cols_m_limit is not n_train:
        #    todo += (n_train - cols_m_limit) * cols_m_limit

        todo, done = K_n_cols, 0
        for done_wkr in pool.imap_unordered(
            partial(
                _assemble_kernel_mat_wkr,
                tril_perms_lin=tril_perms_lin,
                sig=sig,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
                # cols_3n_keep_idxs=cols_3n_keep_idxs,
            ),
            J,
        ):
            done += done_wkr

            if progr_callback is not None and done < todo:
                progr_callback(done, todo)

        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        stop = timeit.default_timer()

        if progr_callback is not None:
            dur_s = (stop - start) / 2
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            progr_callback(todo, todo, sec_disp_str=sec_disp_str)

        # Release some memory.
        glob.pop('K', None)
        glob.pop('R_desc', None)
        glob.pop('R_d_desc', None)

        return np.frombuffer(K).reshape(glob['K_shape'])  # * 1./sig

    def draw_strat_sample(self, T, n, excl_idxs=None):
        """
        Draw sample from dataset that preserves its original distribution.

        The distribution is estimated from a histogram were the bin size is
        determined using the Freedman-Diaconis rule. This rule is designed to
        minimize the difference between the area under the empirical
        probability distribution and the area under the theoretical
        probability distribution. A reduced histogram is then constructed by
        sampling uniformly in each bin. It is intended to populate all bins
        with at least one sample in the reduced histogram, even for small
        training sizes.

        Parameters
        ----------
            T : :obj:`numpy.ndarray`
                Dataset to sample from.
            n : int
                Number of examples.
            excl_idxs : :obj:`numpy.ndarray`, optional
                Array of indices to exclude from sample.

        Returns
        -------
            :obj:`numpy.ndarray`
                Array of indices that form the sample.
        """

        if n == 0:
            return np.array([], dtype=np.uint)

        if T.size == n:  # TODO: this only works if excl_idxs=None
            return np.arange(n)

        if n == 1:
            idxs_all_non_excl = np.setdiff1d(
                np.arange(T.size), excl_idxs, assume_unique=True
            )
            return np.array([np.random.choice(idxs_all_non_excl)])

        # Freedman-Diaconis rule
        h = 2 * np.subtract(*np.percentile(T, [75, 25])) / np.cbrt(n)
        n_bins = int(np.ceil((np.max(T) - np.min(T)) / h)) if h > 0 else 1
        n_bins = min(
            n_bins, int(n / 2)
        )  # Limit number of bins to half of requested subset size.

        bins = np.linspace(np.min(T), np.max(T), n_bins, endpoint=False)
        idxs = np.digitize(T, bins)

        # Exclude restricted indices.
        if excl_idxs is not None and excl_idxs.size > 0:
            idxs[excl_idxs] = n_bins + 1  # Impossible bin.

        uniq_all, cnts_all = np.unique(idxs, return_counts=True)

        # Remove restricted bin.
        if excl_idxs is not None and excl_idxs.size > 0:
            excl_bin_idx = np.where(uniq_all == n_bins + 1)
            cnts_all = np.delete(cnts_all, excl_bin_idx)
            uniq_all = np.delete(uniq_all, excl_bin_idx)

        # Compute reduced bin counts.
        reduced_cnts = np.ceil(cnts_all / np.sum(cnts_all, dtype=float) * n).astype(int)
        reduced_cnts = np.minimum(
            reduced_cnts, cnts_all
        )  # limit reduced_cnts to what is available in cnts_all

        # Reduce/increase bin counts to desired total number of points.
        reduced_cnts_delta = n - np.sum(reduced_cnts)

        while np.abs(reduced_cnts_delta) > 0:

            # How many members can we remove from an arbitrary bucket, without any bucket with more than one member going to zero?
            max_bin_reduction = np.min(reduced_cnts[np.where(reduced_cnts > 1)]) - 1

            # Generate additional bin members to fill up/drain bucket counts of subset. This array contains (repeated) bucket IDs.
            outstanding = np.random.choice(
                uniq_all,
                min(max_bin_reduction, np.abs(reduced_cnts_delta)),
                p=(reduced_cnts - 1) / np.sum(reduced_cnts - 1, dtype=float),
                replace=False,
            )
            uniq_outstanding, cnts_outstanding = np.unique(
                outstanding, return_counts=True
            )  # Aggregate bucket IDs.

            outstanding_bucket_idx = np.where(
                np.in1d(uniq_all, uniq_outstanding, assume_unique=True)
            )[
                0
            ]  # Bucket IDs to Idxs.
            reduced_cnts[outstanding_bucket_idx] += (
                np.sign(reduced_cnts_delta) * cnts_outstanding
            )
            reduced_cnts_delta = n - np.sum(reduced_cnts)

        # Draw examples for each bin.
        idxs_train = np.empty((0,), dtype=int)
        for uniq_idx, bin_cnt in zip(uniq_all, reduced_cnts):
            idx_in_bin_all = np.where(idxs.ravel() == uniq_idx)[0]
            idxs_train = np.append(
                idxs_train, np.random.choice(idx_in_bin_all, bin_cnt, replace=False)
            )
        return idxs_train

    def _solve_closed(self, K, y, lam, callback=None):

        start = timeit.default_timer()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if K.shape[0] == K.shape[1]:

                K[np.diag_indices_from(K)] -= lam  # regularize

                if callback is not None:
                    callback = partial(
                        callback,
                        disp_str='Solving linear system (Cholesky factorization)',
                    )
                    callback(is_done=False)

                try:
                    # Cholesky
                    L, lower = sp.linalg.cho_factor(
                        -K, overwrite_a=True, check_finite=False
                    )
                    alphas = -sp.linalg.cho_solve(
                        (L, lower), y, overwrite_b=True, check_finite=False
                    )
                except np.linalg.LinAlgError:  # try a solver that makes less assumptions

                    if callback is not None:
                        callback = partial(
                            callback,
                            disp_str='Solving linear system (LU factorization)      ',  # Keep whitespaces!
                        )
                        callback(is_done=False)

                    try:
                        # LU
                        alphas = sp.linalg.solve(
                            K, y, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    except MemoryError:
                        self.log.critical(
                            'Not enough memory to train this system using a closed form solver.\n'
                            + 'Please reduce the size of the training set or consider one of the approximate solver options.'
                        )
                        print()
                        sys.exit()

                except MemoryError:
                    self.log.critical(
                        'Not enough memory to train this system using a closed form solver.\n'
                        + 'Please reduce the size of the training set or consider one of the approximate solver options.'
                    )
                    print()
                    sys.exit()
            else:

                if callback is not None:
                    callback = partial(
                        callback,
                        disp_str='Solving overdetermined linear system (least squares approximation)',
                    )
                    callback(is_done=False)

                # least squares for non-square K
                alphas = np.linalg.lstsq(K, y, rcond=-1)[0]

        stop = timeit.default_timer()

        if callback is not None:
            dur_s = (stop - start) / 2
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            callback(is_done=True, sec_disp_str=sec_disp_str)

        return alphas

    def _lev_scores(
        self,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        sig,
        lam,
        use_E_cstr,
        progr_callback=None,
    ):

        n_train, _, dim_i = R_d_desc.shape

        # self.log.info(
        #    'Computing approximate leverage scores for Nystrom preconditioner.'
        # )

        # mem_avail = virtual_memory().available
        # mem_per_point = (
        #     n_train * dim_i ** 2 * 8
        # )  # There are 3 * n_atoms columns per training point.

        # pts_max = int(
        #     np.round((mem_avail * 0.5) / mem_per_point)
        # )  # How many points can be fit into available memory?


        pts_max = 75 # TODO: hardcoded to avoid psutils

        inducing_pts = min(
            min(n_train, pts_max), 75
        )  # How many inducing points to use (for Nystrom approximation, as well as the approximation of leverage scores). Never use more than 75 training points.

        self.log.info(
            '{:d} out of {:d} training points were chosen as support for Nystrom preconditioner.'.format(
                inducing_pts, n_train
            )
        )

        # Convert from training points to actual columns.
        inducing_pts *= dim_i

        # Which columns to use for leverage score approximation?
        # lev_approx_idxs = np.sort(np.random.choice(n_train*dim_i, num_cols_approx, replace=False)) # random subset of columns
        lev_approx_idxs = np.s_[
            :inducing_pts
        ]  # first 'inducing_pts' columns (faster kernel construction)

        if progr_callback is not None:
            progr_callback = partial(
                progr_callback,
                disp_str='Assembling partial kernel matrix (for approx. leverage scores)',
            )

        K_nm = self._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            use_E_cstr=use_E_cstr,
            progr_callback=progr_callback,
            col_idxs=lev_approx_idxs,
        )

        K_mm = K_nm[lev_approx_idxs, :]

        L, lower = self._cho_factor_stable(-K_mm)

        B = sp.linalg.solve_triangular(
            L, -K_nm.T, lower=lower, trans='T', overwrite_b=True, check_finite=False
        )
        # from scipy.linalg.blas import dtrsv
        # B = dtrsv(L, -K_nm.T, lower=lower, trans=1, overwrite_x=1) # same but faster

        B_BT_lam = B.dot(B.T)
        B_BT_lam[np.diag_indices_from(B_BT_lam)] += lam

        # Leverage scores for all columns.
        lev_scores = np.einsum('ij,ij->j', B, np.linalg.solve(B_BT_lam, B))

        # Try to group columns by molecule to speed up kernel matrix generation:
        lev_scores = np.around(
            lev_scores, decimals=1
        )  # Round leverage scores to first decimal place, then sort by score and training point index combined.
        point_idxs = np.tile(np.arange(n_train)[:, None], (1, dim_i)).ravel()
        max_lev_idxs = np.sort(
            np.lexsort((point_idxs, lev_scores))[-inducing_pts:]
        )  # sort by 'lev_scores' then by 'point_idxs'

        # indices of highest 'num_cols_approx' leverage scores
        # max_lev_idxs = np.sort(np.argpartition(lev, -num_cols_approx)[-num_cols_approx:])

        return lev_approx_idxs, max_lev_idxs

    # performs a cholesky decompostion of a matrix, but regularizes the matrix (if neeeded) until its positive definite
    def _cho_factor_stable(self, M, retry_limit=10):
        """
        Performs a Cholesky decompostion of a matrix, but regularizes
        as needed until its positive definite.

        Parameters
        ----------
            M : :obj:`numpy.ndarray`
                Matrix to factorize.
            retry_limit : int, optional
                Limit the number of retries until giving up trying to make the matrix positive definite.

        Returns
        -------
            :obj:`numpy.ndarray`
                Matrix whose upper or lower triangle contains the Cholesky factor of a. Other parts of the matrix contain random data.
            boolean
                Flag indicating whether the factor is in the lower or upper triangle
        """

        n_retries = 0

        retry = True
        while retry:
            try:

                L, lower = sp.linalg.cho_factor(M, overwrite_a=True, check_finite=False)

                n_retries += 1
                retry = False

            except np.linalg.LinAlgError as e:
                if 'not positive definite' in str(e) and n_retries < retry_limit:

                    eps = np.finfo(float).eps
                    lo_eig = sp.linalg.eigh(M, eigvals_only=True, eigvals=(0, 0))
                    if lo_eig < 0:
                        M[np.diag_indices_from(M)] += -lo_eig + eps
                    elif lo_eig < eps:
                        M[np.diag_indices_from(M)] += eps

                    retry = True
                else:
                    raise e

        return L, lower

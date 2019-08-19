"""
This module contains all routines for training GDML and sGDML models.
"""

# MIT License
#
# Copyright (c) 2018-2019 Stefan Chmiela
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

import multiprocessing as mp
import sys
import timeit
import warnings
from functools import partial

import numpy as np
import scipy as sp

from . import __version__
from .predict import GDMLPredict
from .utils import desc, io, perm, ui

glob = {}


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
    j, n_perms, tril_perms_lin, sig, use_E_cstr=False, j_end=None
):  # TODO document j_end
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
        n_perms : int
            Number of individual permutations encoded in
            `tril_perms_lin`.
        tril_perms_lin : :obj:`numpy.ndarray`
            1D array (int) containing all recovered permutations
            expanded as one large permutation to be applied to a tiled
            copy of the object to be permuted.
        sig : int
            Hyper-parameter :math:`\sigma`.

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

    mat52_base_div = 3 * sig ** 4
    sqrt5 = np.sqrt(5.0)
    sig_pow2 = sig ** 2

    base = np.arange(dim_i)  # base set of indices
    blk_j = base + j * dim_i

    E_off = dim_i * n_train

    # Create permutated variants of 'rj_desc' and 'rj_d_desc'.
    rj_desc_perms = np.reshape(
        np.tile(R_desc[j, :], n_perms)[tril_perms_lin], (n_perms, -1), order='F'
    )
    rj_d_desc_perms = np.reshape(
        np.tile(R_d_desc[j, :, :].T, n_perms)[:, tril_perms_lin], (-1, dim_d, n_perms)
    )

    for i in range(j, n_train):

        blk_i = base[:, np.newaxis] + i * dim_i

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
            ((sig_pow2 + sig * norm_ab_perms) * mat52_base_perms),
        )

        # K[blk_i, blk_j] = K[blk_j, blk_i] = R_d_desc[i, :, :].T.dot(diff_ab_outer_perms)
        K[blk_i, blk_j] = R_d_desc[i, :, :].T.dot(diff_ab_outer_perms)
        if (
            i < j_end
        ):  # symmetric extension is not always possible, if a partial kernel is assembled
            K[blk_j, blk_i] = R_d_desc[i, :, :].T.dot(diff_ab_outer_perms)

    if use_E_cstr:
        for i in range(n_train):

            blk_i = base[:, np.newaxis] + i * dim_i

            diff_ab_perms = R_desc[i, :] - rj_desc_perms
            norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

            if use_E_cstr:
                K_fe = (
                    5
                    * diff_ab_perms
                    / (3 * sig ** 3)
                    * (norm_ab_perms[:, None] + sig)
                    * np.exp(-norm_ab_perms / sig)[:, None]
                )
                K[E_off + i, blk_j] = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)

                K[E_off + i, E_off + j] = K[E_off + j, E_off + i] = (
                    1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
                ).dot(np.exp(-norm_ab_perms / sig))

    return n_train - j


class GDMLTrain(object):
    def __init__(self, max_processes=None):
        self._max_processes = max_processes

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
                'No energy labels found in dataset!'
                + '\n       By default, force fields are always reconstructed including the'
                + '\n       corresponding potential energy surface (this can be turned off).\n'
                + '\n       However, the energy labels are missing in the provided dataset.\n'
            )

        use_E_cstr = use_E and use_E_cstr

        ui.progr_toggle(is_done=False, disp_str='Hashing dataset(s)...')
        md5_train = io.dataset_md5(train_dataset)
        md5_valid = io.dataset_md5(valid_dataset)
        ui.progr_toggle(is_done=True, disp_str='Hashing dataset(s)...')

        ui.progr_toggle(
            is_done=False, disp_str='Sampling training and validation subset...'
        )
        if 'E' in train_dataset:
            idxs_train = self.draw_strat_sample(train_dataset['E'], n_train)
        else:
            idxs_train = np.random.choice(
                np.arange(train_dataset['F'].shape[0]), n_train, replace=False
            )

        excl_idxs = idxs_train if md5_train == md5_valid else None
        if 'E' in valid_dataset:
            idxs_valid = self.draw_strat_sample(valid_dataset['E'], n_valid, excl_idxs)
        else:
            idxs_valid_all = np.setdiff1d(
                np.arange(valid_dataset['F'].shape[0]), excl_idxs, assume_unique=True
            )
            idxs_valid = np.random.choice(idxs_valid_all, n_valid, replace=False)
        ui.progr_toggle(
            is_done=True, disp_str='Sampling training and validation subset...'
        )

        R_train = train_dataset['R'][idxs_train, :, :]


        #model = {'R': R_train,}
        #np.savez_compressed('ethanol_200_s22_gdml_R', **model)
        #sys.exit()

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
        }

        if use_E:
            task['E_train'] = train_dataset['E'][idxs_train]

        if 'lattice' in train_dataset:
            task['lattice'] = train_dataset['lattice']

        if use_sym:
            task['perms'] = perm.sync_mat(
                R_train, train_dataset['z'], self._max_processes
            )
            task['perms'] = perm.complete_group(task['perms'])
        else:
            task['perms'] = np.arange(train_dataset['R'].shape[1])[
                None, :
            ]  # no symmetries

        return task

    def train(  # noqa: C901
        self,
        task,
        use_cg,
        cprsn_callback=None,
        ker_progr_callback=None,
        solve_callback=None,
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
                If the provided dataset contains unsupported lattice
                vectors.
        """

        sig = np.squeeze(task['sig'])
        lam = np.squeeze(task['lam'])

        n_perms = task['perms'].shape[0]
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        n_train, n_atoms = task['R_train'].shape[:2]
        dim_i = 3 * n_atoms
        dim_d = tril_perms.shape[1]

        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')

        # check if lattice vectors are supported by this version of the code
        ucell_size = None
        if 'lattice' in task:
            lat = task['lattice']
            if is_lattice_supported(lat):
                ucell_size = np.diag(lat)[0]
            else:
                raise ValueError(
                    'Provided dataset contains unsupported lattice vectors!'
                )

        R_desc = np.empty([n_train, dim_d])
        R_d_desc = np.empty([n_train, dim_d, dim_i])

        for i in range(n_train):
            r = task['R_train'][i]
            pdist = sp.spatial.distance.pdist(r, 'euclidean')

            # pairwise distance with or without minimum-image convention as periodic boundary condition
            if ucell_size is not None:
                pdist = sp.spatial.distance.pdist(
                    r, lambda u, v: np.linalg.norm(desc.pbc_diff(u, v, ucell_size))
                )
            else:
                pdist = sp.spatial.distance.pdist(r, 'euclidean')
            pdist = sp.spatial.distance.squareform(pdist)

            R_desc[i, :] = desc.r_to_desc(r, pdist)
            R_d_desc[i, :, :] = desc.r_to_d_desc(r, pdist, ucell_size)

        if task['use_cprsn'] and n_perms > 1:
            _, cprsn_keep_idxs = np.unique(
                np.sort(task['perms'], axis=0), axis=1, return_index=True
            )

            # _, _, inv_idxs = np.unique(
            #     np.sort(task['perms'], axis=0), axis=1, return_index=True, return_inverse=True
            # )

            # R_d_desc = R_d_desc.reshape(n_train,dim_d,n_atoms,3)
            # task = dict(task)  #
            # for kii,ki in enumerate(cprsn_keep_idxs):
            #     idx_to = ki
            #     idxs_from = np.where(inv_idxs==kii)[0]

            #     for fr in idxs_from[1:]:
            #         R_d_desc[:,:,idx_to,:] += R_d_desc[:,:,fr,:] / len(idxs_from)
            #         task['F_train'][:,idx_to,:] += task['F_train'][:,fr,:] / len(idxs_from)
            # R_d_desc = R_d_desc.reshape(n_train,dim_d,-1)

            cprsn_keep_idxs_lin = (
                np.arange(dim_i).reshape(n_atoms, -1)[cprsn_keep_idxs, :].ravel()
            )

            if cprsn_callback is not None:
                cprsn_callback(n_atoms, cprsn_keep_idxs.shape[0])

            task = dict(task)  # enable item assignment in NPZ
            task['F_train'] = task['F_train'][:, cprsn_keep_idxs, :]
            R_d_desc = R_d_desc[:, :, cprsn_keep_idxs_lin]

        Ft = task['F_train'].ravel()
        Ft_std = np.std(Ft)
        Ft /= Ft_std

        # test

        # n = 0.1

        # test

        # for nystrom precondiner if cg solver is used
        M = int(np.ceil(np.sqrt(n_train))) * 3
        #M = 100

        y = Ft
        if task['use_E'] and task['use_E_cstr']:
            Et = task['E_train'].ravel()
            Et /= Ft_std

            y = np.hstack((Ft, Et))
            # y = np.hstack((n*Ft, (1-n)*Et))

        K = self._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            n_perms,
            tril_perms_lin,
            sig,
            use_E_cstr=task['use_E_cstr'],
            progr_callback=ker_progr_callback,
            j_end=M if use_cg else None,
        )

        # test 2

        # use_ny = True
        # M = 90
        # K_mm = K[:M*R_d_desc.shape[2], :]
        # K_mm = K_mm[:, :M*R_d_desc.shape[2]]
        # K_nm = K[:, :M*R_d_desc.shape[2]]
        # K_mm_mn = np.linalg.lstsq(K_mm,K_nm.T, rcond=-1)[0]
        # K =  K_nm.dot(K_mm_mn)
        # R_d_desc = R_d_desc[:M, :, :]

        # test 2

        if use_cg:

            K_mm = K[: M * dim_i, :]
            K_nm = K

            lam = 1e-8

            # ny_idxs = np.random.choice(K.shape[0], M*R_d_desc.shape[2], replace=False)
            # K_mm = K[ny_idxs, :]
            # K_mm = K_mm[:, ny_idxs]
            # K_nm = K[:, ny_idxs]

            _lup = sp.linalg.lu_factor((-lam) * K_mm + K_nm.T.dot(K_nm))
            def mv(v):
                P_v = -(-1.0 / lam) * (
                    K_nm.dot(sp.linalg.lu_solve(_lup, K_nm.T.dot(v))) - v
                )
                return P_v

            from scipy.sparse.linalg import LinearOperator

            P_op = LinearOperator((n_train * dim_i, n_train * dim_i), matvec=mv)

            global mv_K_first, gdml_predict
            mv_K_first = True
            gdml_predict = None

            def mv_K(v):

                global mv_K_first, gdml_predict

                r_dim = R_d_desc.shape[2]
                r_d_desc_alpha = [
                    rj_d_desc.dot(v[(j * r_dim) : ((j + 1) * r_dim)])
                    for j, rj_d_desc in enumerate(R_d_desc)
                ]

                model = {
                    'type': 'm',
                    'code_version': __version__,
                    'dataset_name': task['dataset_name'],
                    'dataset_theory': task['dataset_theory'],
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
                    'std': 1.0,
                    'sig': sig,
                    'perms': task['perms'],
                    'tril_perms_lin': tril_perms_lin,
                    'use_E': task['use_E'],
                    'use_cprsn': task['use_cprsn'],
                }

                if mv_K_first == True:

                    from .predict import GDMLPredict

                    gdml_predict = GDMLPredict(model)
                    gdml_predict.prepare_parallel(n_bulk=n_train)

                    mv_K_first = False
                else:
                    gdml_predict.set_alphas(r_d_desc_alpha, model)

                R = task['R_train'].reshape(n_train, -1)
                _, f_pred = gdml_predict.predict(R)

                return f_pred.ravel() - lam * v

            K_op = LinearOperator((n_train * dim_i, n_train * dim_i), matvec=mv_K)

        # import matplotlib.pyplot as plt

        # plt.imshow(sp.linalg.solve(K, K2, overwrite_a=False, overwrite_b=False, check_finite=True))
        # plt.imshow(P_inv3.dot(K))
        # plt.imshow(K.dot(np.linalg.inv(K_mm)))
        # plt.colorbar()
        # plt.show()

        # print(np.linalg.inv(K) - np.linalg.inv(K2))

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

        if solve_callback is not None:
            solve_callback(is_done=False)

        start = timeit.default_timer()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if not use_cg:

                K[np.diag_indices_from(K)] -= lam  # regularize

                #model = {'K': K,'F': y}
                #np.savez_compressed('ethanol_200_s22_gdml_KF', **model)
                #sys.exit()

                try:
                    # Cholesky
                    L, lower = sp.linalg.cho_factor(
                        -K, overwrite_a=True, check_finite=False
                    )
                    alphas = -sp.linalg.cho_solve(
                        (L, lower), y, overwrite_b=True, check_finite=False
                    )
                except Exception:
                    # LU
                    alphas = sp.linalg.solve(
                        K, y, overwrite_a=True, overwrite_b=True, check_finite=False
                    )

                # alphas = np.linalg.lstsq(K,Ft)[0]

            else:

                global num_iters

                num_iters = 0

                def callback(xk):
                    global num_iters
                    if num_iters % 10 == 0:

                        solve_callback(
                            is_done=False,
                            sec_disp_str=(
                                '(iter: %d: residual: %.5f)'
                                % (num_iters, np.mean(np.abs(K_op.dot(-xk) - y)))
                            ),
                        )
                    num_iters += 1

                del K

                from scipy.sparse.linalg import cg
                alphas, status = cg(-K_op, y, M=P_op, tol=1e-4, maxiter=3 * n_atoms * n_train, callback=callback)
                alphas = -alphas

        # test 2

        # alphas = K_mm_mn.dot(alphas)  # remove me later

        # test 2

        stop = timeit.default_timer()

        alphas_F = alphas
        if task['use_E_cstr']:
            alphas_E = alphas[-n_train:]
            alphas_F = alphas[:-n_train]

        if solve_callback is not None:

            sec_disp_str = (
                '(%d: residual: %.5f)' % (num_iters, np.mean(np.abs(K_op.dot(alphas) - y)))
                if use_cg
                else '(%.1f s)' % ((stop - start) / 2)
            )
            solve_callback(is_done=True, sec_disp_str=sec_disp_str)

        r_dim = R_d_desc.shape[2]
        r_d_desc_alpha = [
            rj_d_desc.dot(alphas_F[(j * r_dim) : ((j + 1) * r_dim)])
            for j, rj_d_desc in enumerate(R_d_desc)
        ]

        # test 2

        # task = dict(task)
        # task['idxs_train'] = task['idxs_train'][:M] # remove me
        # R_desc = R_desc[:M,:] # remove me

        # test 2

        model = {
            'type': 'm',
            'code_version': __version__,
            'dataset_name': task['dataset_name'],
            'dataset_theory': task['dataset_theory'],
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
            'std': Ft_std,
            'sig': sig,
            'perms': task['perms'],
            'tril_perms_lin': tril_perms_lin,
            'use_E': task['use_E'],
            'use_cprsn': task['use_cprsn'],
        }

        if task['use_E']:
            model['e_err'] = {'mae': np.nan, 'rmse': np.nan}

            if task['use_E_cstr']:
                model['alphas_E'] = alphas_E
            else:
                model['c'] = self._recov_int_const(model, task)

        if 'lattice' in task:
            model['lattice'] = task['lattice']

        return model

    def _recov_int_const(self, model, task):
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

        gdml = GDMLPredict(model)
        n_train = task['E_train'].shape[0]

        R = task['R_train'].reshape(n_train, -1)

        E_pred, _ = gdml.predict(R)
        E_ref = np.squeeze(task['E_train'])

        # E_ref = E_ref[0] # debug remove me NEW

        # _,F_pred = gdml.predict(R)
        # print
        # print task['F_train'].shape
        # print E_pred + np.sum(E_ref - E_pred) / E_ref.shape[0]
        # print E_ref

        # import matplotlib.pyplot as plt
        # plt.plot(E_ref, '--', label='ref')
        # plt.plot(E_pred + np.sum(E_ref - E_pred) / E_ref.shape[0], label='pred')
        # plt.plot(E_pred, label='pred')
        # plt.title('energy')
        # plt.legend()
        # plt.show()

        # plt.plot(task['F_train'][:,0,2] , '--', label='ref')
        # plt.plot(F_pred[:,2], label='pred')
        # plt.title('force')
        # plt.legend()
        # plt.show()

        e_fact = np.linalg.lstsq(
            np.column_stack((E_pred, np.ones(E_ref.shape))), E_ref, rcond=-1
        )[0][0]
        corrcoef = np.corrcoef(E_ref, E_pred)[0, 1]

        if np.sign(e_fact) == -1:
            raise ValueError(
                'Provided dataset contains gradients instead of force labels (flipped sign). Please correct!'
            )

        if corrcoef < 0.95:
            raise ValueError(
                'Inconsistent energy labels detected!'
                + '\n       The predicted energies for the training data are only weakly correlated'
                + '\n       with the reference labels (correlation coefficient %.2f) which indicates'
                % corrcoef
                + '\n       that the issue is most likely NOT just a unit conversion error.\n'
                + '\n       Troubleshooting tips:'
                + '\n         (1) Verify correct correspondence between geometries and labels in'
                + '\n             the provided dataset.'
                + '\n         (2) Verify consistency between energy and force labels.'
                + '\n               - Correspondence correct?'
                + '\n               - Same level of theory?'
                + '\n               - Accuracy of forces (if numerical)?'
                + '\n         (3) Is the training data spread too broadly (i.e. weakly sampled'
                + '\n             transitions between example clusters)?'
                + '\n         (4) Are there duplicate geometries in the training data?'
                + '\n         (5) Are there any corrupted data points (e.g. parsing errors)?\n'
            )

        if np.abs(e_fact - 1) > 1e-1:
            raise ValueError(
                'Different scales in energy vs. force labels detected!'
                + '\n       The integrated forces differ from energy labels by factor ~%.2E.\n'
                % e_fact
                + '\n       Troubleshooting tips:'
                + '\n         (1) Verify consistency of units in energy and force labels.'
                + '\n         (2) Is the training data spread too broadly (i.e. weakly sampled'
                + '\n             transitions between example clusters)?\n'
            )

            # Least squares estimate for integration constant.
        return np.sum(E_ref - E_pred) / E_ref.shape[0]


    def _assemble_kernel_mat(
        self,
        R_desc,
        R_d_desc,
        n_perms,
        tril_perms_lin,
        sig,
        use_E_cstr=False,
        progr_callback=None,
        j_end=None,
    ):  # TODO: document j_end
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
            n_perms : int
                Number of individual permutations encoded in
                `tril_perms_lin`.
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

        Returns
        -------
            :obj:`numpy.ndarray`
                Force field kernel matrix.
        """

        global glob

        n_train, dim_d, dim_i = R_d_desc.shape

        if j_end == None:
            j_end = n_train

        dim_K = n_train * dim_i
        dim_K_ny = j_end * dim_i
        dim_K += n_train if use_E_cstr else 0

        K = mp.RawArray('d', dim_K_ny * dim_K)
        glob['K'], glob['K_shape'] = K, (dim_K, dim_K_ny)
        glob['R_desc'], glob['R_desc_shape'] = _share_array(R_desc, 'd')
        glob['R_d_desc'], glob['R_d_desc_shape'] = _share_array(R_d_desc, 'd')

        start = timeit.default_timer()

        pool = mp.Pool(self._max_processes)

        # todo = (n_train ** 2 - n_train) // 2 + n_train
        todo = (j_end ** 2 - j_end) // 2 + j_end

        if j_end is not n_train:
            todo += (n_train - j_end) * j_end

        done_total = 0
        for done in pool.imap_unordered(
            partial(
                _assemble_kernel_mat_wkr,
                n_perms=n_perms,
                tril_perms_lin=tril_perms_lin,
                sig=sig,
                use_E_cstr=use_E_cstr,
                j_end=j_end,
            ),
            list(range(j_end)),
        ):
            done_total += done

            if progr_callback is not None:
                if done_total == todo:
                    stop = timeit.default_timer()
                    progr_callback(done_total, todo, sec_disp_str='(%.1f s)' % ((stop - start) / 2))
                else:
                    progr_callback(done_total, todo)

        pool.close()

        # Release some memory.
        glob.pop('K', None)
        glob.pop('R_desc', None)
        glob.pop('R_d_desc', None)

        return np.frombuffer(K).reshape(glob['K_shape'])

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

        if T.size == n:
            return np.arange(n)

        # Freedman-Diaconis rule
        h = 2 * np.subtract(*np.percentile(T, [75, 25])) / np.cbrt(n)
        n_bins = int(np.ceil((np.max(T) - np.min(T)) / h)) if h > 0 else 1
        n_bins = min(
            n_bins, n / 2
        )  # Limit number of bins to half of requested subset size.

        bins = np.linspace(np.min(T), np.max(T), n_bins, endpoint=False)
        idxs = np.digitize(T, bins)

        # Exclude restricted indices.
        if excl_idxs is not None:
            idxs[excl_idxs] = n_bins + 1  # Impossible bin.

        uniq_all, cnts_all = np.unique(idxs, return_counts=True)

        # Remove restricted bin.
        if excl_idxs is not None:
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

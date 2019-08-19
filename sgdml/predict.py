"""
This module contains all routines for evaluating GDML and sGDML models.
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

import os
import multiprocessing as mp
import timeit
from functools import partial

import numpy as np
import scipy.spatial.distance

from . import __version__
from .utils import desc

glob = {}


def share_array(arr_np):
    """
    Return a ctypes array allocated from shared memory with data from a
    NumPy array of type `float`.

    Parameters
    ----------
            arr_np : :obj:`numpy.ndarray`
                    NumPy array.

    Returns
    -------
            array of :obj:`ctype`
    """

    arr = mp.RawArray('d', arr_np.ravel())
    return arr, arr_np.shape


def _predict(r, n_train, std, c, chunk_size, ucell_size):

    r = r.reshape(-1, 3)

    if ucell_size is None:
        pdist = scipy.spatial.distance.pdist(r, 'euclidean')
    else:
        pdist = scipy.spatial.distance.pdist(
            r, lambda u, v: np.linalg.norm(desc.pbc_diff(u, v, ucell_size))
        )
    pdist = scipy.spatial.distance.squareform(pdist, checks=False)

    r_desc = desc.r_to_desc(r, pdist)
    # r_d_desc = desc.r_to_d_desc(r, pdist)

    res = _predict_wkr((0, n_train), chunk_size, r_desc)
    res *= std

    F = desc.r_to_d_desc_op(r, pdist, res[1:], ucell_size).reshape(1, -1)
    # F = res[1:].reshape(1,-1).dot(r_d_desc)
    return (res[0] + c).reshape(-1), F


def _predict_wkr(wkr_start_stop, chunk_size, r_desc):
    """
    Compute part of a prediction.

    The workload will be processed in `b_size` chunks.

    Parameters
    ----------
            wkr_start_stop : tuple of int
                    Indices of first and last (exclusive) sum element.
            r_desc : :obj:`numpy.ndarray`
                    1D array containing the descriptor for the query
                    geometry.

    Returns
    -------
            :obj:`numpy.ndarray`
                    Partial prediction of all force components and
                    energy (appended to array as last element).
    """

    global glob, sig, n_perms

    wkr_start, wkr_stop = wkr_start_stop

    R_desc_perms = np.frombuffer(glob['R_desc_perms']).reshape(
        glob['R_desc_perms_shape']
    )
    R_d_desc_alpha_perms = np.frombuffer(glob['R_d_desc_alpha_perms']).reshape(
        glob['R_d_desc_alpha_perms_shape']
    )

    if 'alphas_E_lin' in glob:
        alphas_E_lin = np.frombuffer(glob['alphas_E_lin']).reshape(
            glob['alphas_E_lin_shape']
        )

    dim_d = r_desc.shape[0]
    dim_c = chunk_size * n_perms

    # pre-allocation

    diff_ab_perms = np.empty((dim_c, dim_d))
    a_x2 = np.empty((dim_c,))
    mat52_base = np.empty((dim_c,))

    mat52_base_fact = 5.0 / (3 * sig ** 3)
    diag_scale_fact = 5.0 / sig
    sqrt5 = np.sqrt(5.0)

    E_F = np.zeros((dim_d + 1,))
    F = E_F[1:]

    wkr_start *= n_perms
    wkr_stop *= n_perms

    b_start = wkr_start
    for b_stop in list(range(wkr_start + dim_c, wkr_stop, dim_c)) + [wkr_stop]:

        rj_desc_perms = R_desc_perms[b_start:b_stop, :]
        rj_d_desc_alpha_perms = R_d_desc_alpha_perms[b_start:b_stop, :]

        # Resize pre-allocated memory for last iteration, if chunk_size is not a divisor of the training set size.
        # Note: It's faster to process equally sized chunks.
        c_size = b_stop - b_start
        if c_size < dim_c:
            diff_ab_perms = diff_ab_perms[:c_size, :]
            a_x2 = a_x2[:c_size]
            mat52_base = mat52_base[:c_size]

        # diff_ab_perms = r_desc - rj_desc_perms
        np.subtract(
            np.broadcast_to(r_desc, rj_desc_perms.shape),
            rj_desc_perms,
            out=diff_ab_perms,
        )
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

        # mat52_base = np.exp(-norm_ab_perms / sig) * mat52_base_fact
        np.exp(-norm_ab_perms / sig, out=mat52_base)
        mat52_base *= mat52_base_fact
        # a_x2 = np.einsum('ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms) # colum wise dot product
        np.einsum(
            'ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms, out=a_x2
        )  # colum wise dot product
        # a_x2 = np.einsum('ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms) * mat52_base # colum wise dot product

        # F += np.linalg.multi_dot([a_x2 * mat52_base, diff_ab_perms, r_d_desc]) * diag_scale_fact
        F += (a_x2 * mat52_base).dot(diff_ab_perms) * diag_scale_fact
        # F += a_x2.dot(diff_ab_perms) * diag_scale_fact
        mat52_base *= norm_ab_perms + sig

        # F -= np.linalg.multi_dot([mat52_base, rj_d_desc_alpha_perms, r_d_desc])
        F -= mat52_base.dot(rj_d_desc_alpha_perms)
        E_F[0] += a_x2.dot(mat52_base)  # this one
        # E_F[0] += np.sum(a_x2)

        if 'alphas_E_lin' in glob:

            K_fe = diff_ab_perms * mat52_base[:, None]
            F += alphas_E_lin[b_start:b_stop].dot(K_fe)

            K_ee = (
                1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            ) * np.exp(-norm_ab_perms / sig)
            E_F[0] += K_ee.dot(alphas_E_lin[b_start:b_stop])

        b_start = b_stop

    return E_F


class GDMLPredict(object):
    def __init__(
        self, model, batch_size=None, num_workers=1, max_processes=None, use_torch=False
    ):
        """
        Query trained sGDML force fields.

        This class is used to load a trained model and make energy and
        force predictions for new geometries. GPU support is provided
        through PyTorch (requires optional `torch` dependency to be
        installed).

        Note
        ----
                The parameters `batch_size` and `num_workers` are only
                relevant if this code runs on a CPU. Both can be set
                automatically via the function
                `prepare_parallel`. Enabling
                calculations via PyTorch is only recommended with GPU
                support. CPU calcuations are faster with our NumPy
                implementation.

        Parameters
        ----------
                model : :obj:`dict`
                        Data structure that holds all parameters of the
                        trained model. This object is the output of
                        `GDMLTrain.train`
                batch_size : int, optional
                        Chunk size for processing parallel tasks.
                num_workers : int, optional
                        Number of parallel workers.
                max_processes : int, optional
                        Limit the max. number of processes. Otherwise
                        all CPU cores are used. This parameters has no
                        effect if `use_torch=True`
                use_torch : boolean, optional
                        Use PyTorch to calculate predictions

        Returns
        -------
                :obj:`numpy.ndarray`
                        Energies stored in an 1D array of size M.
                :obj:`numpy.ndarray`
                        Forces stored in an 2D arry of size M x 3N.
        """

        global glob, sig, n_perms

        self.n_atoms = model['z'].shape[0]

        self.ucell_size = lat_diag[0] if 'lattice' in model else None

        self.n_train = model['R_desc'].shape[1]
        sig = model['sig']

        self.std = model['std'] if 'std' in model else 1.0
        self.c = model['c']

        n_perms = model['perms'].shape[0]

        # Precompute permuted training descriptors and its first derivatives multiplied with the coefficients (only needed for cached variant).
        R_desc_perms = np.reshape(
            np.tile(model['R_desc'].T, n_perms)[:, model['tril_perms_lin']],
            (self.n_train * n_perms, -1),
            order='F',
        )
        R_desc_perms = np.swapaxes(
            R_desc_perms.reshape(n_perms, self.n_train, -1), 0, 1
        ).reshape((self.n_train * n_perms, -1))
        glob['R_desc_perms'], glob['R_desc_perms_shape'] = share_array(R_desc_perms)

        R_d_desc_alpha_perms = np.reshape(
            np.tile(model['R_d_desc_alpha'], n_perms)[:, model['tril_perms_lin']],
            (self.n_train * n_perms, -1),
            order='F',
        )
        R_d_desc_alpha_perms = np.swapaxes(
            R_d_desc_alpha_perms.reshape(n_perms, self.n_train, -1), 0, 1
        ).reshape((self.n_train * n_perms, -1))
        glob['R_d_desc_alpha_perms'], glob['R_d_desc_alpha_perms_shape'] = share_array(
            R_d_desc_alpha_perms
        )

        if 'alphas_E' in model:
            alphas_E_lin = np.tile(model['alphas_E'][:, None], (1, n_perms)).ravel()
            glob['alphas_E_lin'], glob['alphas_E_lin_shape'] = share_array(alphas_E_lin)

        # GPU support

        self.use_torch = use_torch
        self.torch_predict = None
        if self.use_torch:
            try:
                import torch
            except ImportError:  # dependency missing, issue a warning
                raise ValueError(
                    'PyTorch calculations requested, without having optional PyTorch dependency installed!'
                    + '\n       Please \'pip install torch\' or disable PyTorch calculations.'
                )

            from .torchtools import GDMLTorchPredict

            self.torch_predict = GDMLTorchPredict(model)

        # Parallel processing configuration

        self._bulk_mp = False  # Bulk predictions with multiple processes?

        # How many parallel processes?
        self._max_processes = max_processes
        if self._max_processes is None:
            self._max_processes = mp.cpu_count()
        self.pool = None
        self._num_workers = 1
        self._set_num_workers(num_workers)

        # Size of chunks in which each parallel task will be processed (unit: number of training samples)
        # This parameter should be as large as possible, but it depends on the size of available memory.
        self._set_batch_size(batch_size)

    def __del__(self):
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.terminate()


    ## Public ##

    def set_alphas(self, R_d_desc_alpha, model): # TODO: document me, fix lazy requirement of parameters

        R_d_desc_alpha_perms = np.reshape(
            np.tile(R_d_desc_alpha, n_perms)[:, model['tril_perms_lin']],
            (self.n_train * n_perms, -1),
            order='F',
        )
        R_d_desc_alpha_perms = np.swapaxes(
            R_d_desc_alpha_perms.reshape(n_perms, self.n_train, -1), 0, 1
        ).reshape((self.n_train * n_perms, -1))
        glob['R_d_desc_alpha_perms'], glob['R_d_desc_alpha_perms_shape'] = share_array(
            R_d_desc_alpha_perms
        )

        self._reset_mp()


    def _set_num_workers(
        self, num_workers=None
    ):  # TODO: complain if chunk or worker parameters do not fit training data (this causes issues with the caching)!!
        """
        Set number of processes to use during prediction.

        If bulk_mp == True, each worker handles the whole generation of single prediction (this if for querying multiple geometries at once)
        If bulk_mp == False, each worker may handle only a part of a prediction (chunks are defined in 'wkr_starts_stops'). In that scenario multiple proesses
        are used to distribute the work of generating a single prediction

        This number should not exceed the number of available CPU cores.

        Note
        ----
                This parameter can be optimally determined using
                `prepare_parallel`.

        Parameters
        ----------
                num_workers : int, optional
                        Number of processes (maximum value is set if
                        `None`).
        """

        if self._num_workers is not num_workers:

            if self.pool is not None:
                self.pool.terminate()
                self.pool.join()
                self.pool = None

            self._num_workers = 1
            if num_workers is None or num_workers > 1:
                self.pool = mp.Pool(processes=num_workers)
                self._num_workers = self.pool._processes

        # Data ranges for processes
        if self._bulk_mp:
            wkr_starts = [self.n_train]
        else:
            wkr_starts = list(
                range(
                    0, self.n_train, int(np.ceil(float(self.n_train) / self._num_workers))
                )
            )
        wkr_stops = wkr_starts[1:] + [self.n_train]

        self.wkr_starts_stops = list(zip(wkr_starts, wkr_stops))


    def _reset_mp(self):

        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

        self.pool = mp.Pool(processes=self._num_workers)
        self._num_workers = self.pool._processes


    def _set_batch_size(
        self, batch_size=None
    ):  # TODO: complain if chunk or worker parameters do not fit training data (this causes issues with the caching)!!
        """
        Set chunk size for each process.

        The chunk size determines how much of a processes workload will
        be passed to Python's underlying low-level routines at once.
        This parameter is highly hardware dependent. A chunk is a subset
        of the training set of the model.

        Note
        ----
                This parameter can be optimally determined using
                `prepare_parallel`.

        Parameters
        ----------
                batch_size : int
                        Chunk size (maximum value is set if `None`).
        """

        if batch_size is None:
            batch_size = self.n_train

        self._chunk_size = batch_size


    def _set_bulk_mp(
        self, bulk_mp = False
    ): 

        if self._bulk_mp is not bulk_mp:
            self._bulk_mp = bulk_mp

            # Reset data ranges for processes stored in 'wkr_starts_stops'
            self._set_num_workers(self._num_workers)


    def set_opt_num_workers_and_batch_size_fast(self, n_bulk=1, n_reps=1): # deprecated
        """
        Warning
        -------
        Deprecated! Please use the function `prepare_parallel` in future projects.
        
        Parameters
        ----------
                n_bulk : int, optional
                        Number of geometries that will be passed to the
                        `predict` function in each call (performance
                        will be optimized for that exact use case).
                n_reps : int, optional
                        Number of repetitions (bigger value: more
                        accurate, but also slower).

        Returns
        -------
                int
                        Force and energy prediciton speed in geometries
                        per second.
        """

        self.prepare_parallel(n_bulk, n_reps)


    def prepare_parallel(self, n_bulk=1, n_reps=1, return_is_from_cache=False):  # noqa: C901
        """
        Find and set the optimal parallelization parameters for the
        currently loaded model, running on a particular system. The result
        also depends on the number of geometries `n_bulk` that will be
        passed at once when calling the `predict` function.

        This function runs a benchmark in which the prediction routine is
        repeatedly called `n_reps`-times (default: 1) with varying parameter
        configurations, while the runtime is measured for each one. The
        optimal parameters are then cached for fast retrival in future
        calls of this function.

        We recommend calling this function after initialization of this
        class, as it will drastically increase the performance of the
        `predict` function.

        Note
        ----
                Depending on the parameter `n_reps`, this routine may take
                some seconds/minutes to complete. However, once a
                statistically significant number of benchmark results has
                been gathered for a particular configuration, it starts
                returning almost instantly.

        Parameters
        ----------
                n_bulk : int, optional
                        Number of geometries that will be passed to the
                        `predict` function in each call (performance
                        will be optimized for that exact use case).
                n_reps : int, optional
                        Number of repetitions (bigger value: more
                        accurate, but also slower).
                return_is_from_cache : bool, optional
                        If enabled, this function returns a second value
                        indicating if the returned results were obtained
                        from cache.
    
        Returns
        -------
                int
                        Force and energy prediciton speed in geometries
                        per second.
                boolean, optional
                        Return, whether this function obtained the results
                        from cache. 
        """

        global n_perms

        # Retrieve cached benchmark results, if available.
        bmark_result = self._load_cached_bmark_result(n_bulk)
        if bmark_result is not None:

            num_workers, batch_size, bulk_mp, gps = bmark_result

            self._set_batch_size(batch_size)
            self._set_num_workers(num_workers)
            self._set_bulk_mp(bulk_mp)

            if return_is_from_cache:
                is_from_cache = True
                return gps, is_from_cache
            else:
                return gps

        best_results = []
        last_i = None

        best_gps = 0
        gps_min = 0.0

        best_params = 1, 1

        #reps_done = 0
        r_dummy = np.random.rand(n_bulk, self.n_atoms * 3)
        def _dummy_predict():
            self.predict(r_dummy)
            # reps_done += 1
            # print(reps_done)

        bulk_mp_rng = [True, False] if n_bulk > 1 else [False]
        for bulk_mp in bulk_mp_rng:
            #self._bulk_mp = bulk_mp
            self._set_bulk_mp(bulk_mp)

            if bulk_mp is False:
                last_i = 0

            num_workers_rng = (
                list(range(self._max_processes, 1, -1))
                if bulk_mp
                else list(range(1, self._max_processes + 1))
            )

            # num_workers_rng_sizes = [batch_size for batch_size in batch_size_rng if min_batch_size % batch_size == 0]

            # for num_workers in range(min_num_workers,self._max_processes+1):
            for num_workers in num_workers_rng:
                if not bulk_mp and self.n_train % num_workers != 0:
                    continue
                    # if bulk_mp and n_bulk % num_workers != 0:
                    # 	continue
                self._set_num_workers(num_workers)

                best_gps = 0
                gps_rng = (np.inf, 0.0)

                min_batch_size = (
                    min(self.n_train, n_bulk)
                    if bulk_mp
                    else int(np.ceil(self.n_train / num_workers))
                )
                batch_size_rng = list(range(min_batch_size, 0, -1))

                # for i in range(0,min_batch_size):
                batch_size_rng_sizes = [
                    batch_size
                    for batch_size in batch_size_rng
                    if min_batch_size % batch_size == 0
                ]

                #print('batch_size_rng_sizes ' + str(bulk_mp))
                #print(batch_size_rng_sizes)

                i_done = 0
                i_dir = 1
                i = 0 if last_i is None else last_i
                # i = 0
                while i >= 0 and i < len(batch_size_rng_sizes):

                    batch_size = batch_size_rng_sizes[i]
                    self._set_batch_size(batch_size)

                    i_done += 1

                    gps = (
                        n_bulk
                        * n_reps
                        / (timeit.timeit(_dummy_predict, number=n_reps))
                    )

                    #print(
                    #    '{:2d}@{:d} {:d} | {:7.2f} gps'.format(
                    #        num_workers, batch_size, bulk_mp, gps
                    #    )
                    #)

                    #print(batch_size * self.n_atoms * n_perms)

                    gps_rng = min(gps_rng[0], gps), max(gps_rng[1], gps)

                    # gps still going up?
                    # AND: gps not lower than the lowest overall?
                    if gps < best_gps and gps >= gps_min:
                        if (
                            i_dir > 0
                            and i_done == 2
                            and batch_size != batch_size_rng_sizes[1]
                        ):  # do we turn?
                            i -= 2 * i_dir
                            i_dir = -1
                            #print('><')
                            continue
                        else:
                            # if batch_size == batch_size_rng_sizes[1]:
                            # 	i -= 1*i_dir
                            #print('>>break ' + str(i_done))
                            break
                    else:
                        best_gps = gps
                        best_params = num_workers, batch_size, bulk_mp

                        # if gps < best_gps:
                        # 	break
                        # else:
                        # 	best_gps = gps
                        # 	best_params = num_workers, batch_size, bulk_mp

                    if (
                        not bulk_mp and n_bulk > 1
                    ):  # stop search early when multiple cpus are available and the 1 cpu case is tested
                        if (
                            gps < gps_min
                        ):  # if the batch size run is lower than the lowest overall, stop right here
                            break

                    i += 1 * i_dir

                last_i = i - 1 * i_dir
                i_dir = 1

                if len(best_results) > 0:
                    overall_best_gps = max(best_results, key=lambda x: x[1])[1]
                    if best_gps < overall_best_gps:
                        break

                    if best_gps < gps_min:
                        break

                gps_min = gps_rng[0]
                # print ('gps_min ' + str(gps_min))

                # print ('best_gps')
                # print (best_gps)

                if len(best_results) > 0 and best_gps < overall_best_gps:
                    break

                best_results.append((best_params, best_gps))

        (num_workers, batch_size, bulk_mp), gps = max(best_results, key=lambda x: x[1])

        # Cache benchmark results.
        self._save_cached_bmark_result(n_bulk, num_workers, batch_size, bulk_mp, gps)

        self._set_batch_size(batch_size)
        self._set_num_workers(num_workers)
        self._set_bulk_mp(bulk_mp)

        if return_is_from_cache:
            is_from_cache = False
            return gps, is_from_cache
        else:
            return gps


    def _save_cached_bmark_result(self, n_bulk, num_workers, batch_size, bulk_mp, gps): # document me

        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        bmark_file = '_bmark_cache.npz'
        bmark_path = os.path.join(pkg_dir, bmark_file)

        bkey = '{}-{}-{}'.format(self.n_train, n_bulk, self._max_processes)

        if os.path.exists(bmark_path):

            with np.load(bmark_path, allow_pickle=True) as bmark:
                bmark = dict(bmark)

                bmark['runs']         = np.append(bmark['runs'], bkey)
                bmark['num_workers']  = np.append(bmark['num_workers'], num_workers)
                bmark['batch_size']   = np.append(bmark['batch_size'], batch_size)
                bmark['bulk_mp']      = np.append(bmark['bulk_mp'], bulk_mp)
                bmark['gps']          = np.append(bmark['gps'], gps)
        else:
            bmark = {
                'code_version': __version__,
                'runs': [bkey],
                'gps': [gps],
                'num_workers': [num_workers],
                'batch_size': [batch_size],
                'bulk_mp': [bulk_mp],
                }

        np.savez_compressed(bmark_path, **bmark)


    def _load_cached_bmark_result(self, n_bulk): # document me

        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        bmark_file = '_bmark_cache.npz'
        bmark_path = os.path.join(pkg_dir, bmark_file)

        bkey = '{}-{}-{}'.format(self.n_train, n_bulk, self._max_processes)

        if not os.path.exists(bmark_path):
            return None

        with np.load(bmark_path, allow_pickle=True) as bmark:

            # Keep collecting benchmark runs, until we have at least three.
            run_idxs = np.where(bmark['runs'] == bkey)[0]
            if len(run_idxs) >= 3:

                config_keys = []
                #print()
                for run_idx in run_idxs:
                   #print(str(B['num_workers'][run_idx]) + '  ' + str(B['batch_size'][run_idx]) + '  ' + str(B['gps'][run_idx]))
                    config_keys.append('{}-{}-{}'.format(bmark['num_workers'][run_idx], bmark['batch_size'][run_idx], bmark['bulk_mp'][run_idx]))

                values, uinverse = np.unique(config_keys, return_index=True)

                best_mean = -1
                best_gps = 0
                for i,config_key in enumerate(zip(values,uinverse)):
                    mean_gps = np.mean(bmark['gps'][np.where(np.array(config_keys) == config_key[0])[0]])

                    if best_gps == 0 or best_gps < mean_gps:
                        best_mean = i
                        best_gps = mean_gps

                best_idx = run_idxs[uinverse[best_mean]]
                num_workers = bmark['num_workers'][best_idx]
                batch_size  = bmark['batch_size'][best_idx]
                bulk_mp     = bmark['bulk_mp'][best_idx]

                return num_workers,batch_size,bulk_mp,best_gps

        return None


    def _predict_bulk(self, R):
        """
        Predict energy and forces for multiple geometries.

        Parameters
        ----------
                R : :obj:`numpy.ndarray`
                        A 2D array of size M x 3N containing of the
                        Cartesian coordinates of each atom of M
                        molecules.

        Returns
        -------
                :obj:`numpy.ndarray`
                        Energies stored in an 1D array of size M.
                :obj:`numpy.ndarray`
                        Forces stored in an 2D arry of size M x 3N.
        """

        n_pred, dim_i = R.shape

        F = np.empty((n_pred, dim_i))
        E = np.empty((n_pred,))

        if self._bulk_mp is True:
            for i, E_F in enumerate(
                self.pool.imap(
                    partial(
                        _predict,
                        n_train=self.n_train,
                        std=self.std,
                        c=self.c,
                        chunk_size=self._chunk_size,
                        ucell_size=self.ucell_size,
                    ),
                    R,
                )
            ):
                E[i], F[i, :] = E_F
        else:
            for i, r in enumerate(R):
                E[i], F[i, :] = self.predict(r)

        return E, F


    def predict(self, r):
        """
        Predict energy and forces for multiple geometries. This function
        can run on the GPU, if the optional PyTorch dependency is
        installed and `use_torch=True` was speciefied during
        initialization of this class.

        Note
        ----
                The order of the atoms in `r` is not arbitrary and must
                be the same as used for training the model.

        Parameters
        ----------
                r : :obj:`numpy.ndarray`
                        A 2D array of size M x 3N containing of the
                        Cartesian coordinates of each atom of M
                        molecules.

        Returns
        -------
                :obj:`numpy.ndarray`
                        Energies stored in an 1D array of size M.
                :obj:`numpy.ndarray`
                        Forces stored in an 2D arry of size M x 3N.
        """

        if self.use_torch:

            import torch

            # hack: add singleton dimension if input is (,3N)
            if r.ndim == 1:
                r = r[None, :]

            M = r.shape[0]

            Rs = torch.from_numpy(r.reshape(M, -1, 3))
            e_pred, f_pred = self.torch_predict.forward(Rs)

            E = e_pred.numpy()
            F = f_pred.numpy().reshape(M, -1)

            return E, F

        if r.ndim == 2 and r.shape[0] > 1:
            return self._predict_bulk(r)

        r = r.reshape(self.n_atoms, 3)
        if self.ucell_size is None:
            pdist = scipy.spatial.distance.pdist(r, 'euclidean')
        else:
            pdist = scipy.spatial.distance.pdist(
                r, lambda u, v: np.linalg.norm(desc.pbc_diff(u, v, self.ucell_size))
            )
        pdist = scipy.spatial.distance.squareform(pdist, checks=False)

        r_desc = desc.r_to_desc(r, pdist)
        # r_d_desc = desc.r_to_d_desc(r, pdist)

        if self._num_workers == 1 or self._bulk_mp:
            res = _predict_wkr((0, self.n_train), self._chunk_size, r_desc)
        else:
            res = sum(
                self.pool.map(
                    partial(_predict_wkr, chunk_size=self._chunk_size, r_desc=r_desc),
                    self.wkr_starts_stops,
                )
            )
        res *= self.std

        E = res[0].reshape(-1) + self.c
        F = desc.r_to_d_desc_op(r, pdist, res[1:], self.ucell_size).reshape(1, -1)
        # F = res[1:].reshape(1,-1).dot(r_d_desc)
        return E, F

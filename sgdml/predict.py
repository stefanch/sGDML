"""
This module contains all routines for evaluating GDML and sGDML models.
"""

# MIT License
#
# Copyright (c) 2018-2020 Stefan Chmiela, Gregory Fonseca
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
import os
import multiprocessing as mp
import timeit
from functools import partial

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True

import numpy as np

from . import __version__
from .utils.desc import Desc


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


def _predict_wkr(
    r, r_desc_d_desc, lat_and_inv, glob_id, wkr_start_stop=None, chunk_size=None
):
    """
    Compute (part) of a prediction.

    Every prediction is a linear combination involving the training points used for
    this model. This function evalutates that combination for the range specified by
    `wkr_start_stop`. This workload can optionally be processed in chunks,
    which can be faster as it requires less memory to be allocated.

    Note
    ----
        It is sufficient to provide either the parameter `r` or `r_desc_d_desc`.
        The other one can be set to `None`.

    Parameters
    ----------
            r : :obj:`numpy.ndarray`
                    An array of size 3N containing the Cartesian
                    coordinates of each atom in the molecule.
            r_desc_d_desc : tuple of :obj:`numpy.ndarray`
                    A tuple made up of:
                        (1) An array of size D containing the descriptors
                        of dimension D for the molecule.
                        (2) An array of size D x 3N containing the
                        descriptor Jacobian for the molecules. It has dimension
                        D with 3N partial derivatives with respect to the 3N
                        Cartesian coordinates of each atom.
            lat_and_inv : tuple of :obj:`numpy.ndarray`
                    Tuple of 3 x 3 matrix containing lattice vectors as columns and
                    its inverse.
            glob_id : int
                    Identifier of the global namespace that this
                    function is supposed to be using (zero if only one
                    instance of this class exists at the same time).
            wkr_start_stop : tuple of int, optional
                    Range defined by the indices of first and last (exclusive)
                    sum element. The full prediction is generated if this parameter
                    is not specified.
            chunk_size : int, optional
                    Chunk size. The whole linear combination is evaluated in a large
                    vector operation instead of looping over smaller chunks if this
                    parameter is left unspecified.

    Returns
    -------
            :obj:`numpy.ndarray`
                    Partial prediction of all force components and
                    energy (appended to array as last element).
    """

    global globs
    glob = globs[glob_id]
    sig, n_perms = glob['sig'], glob['n_perms']

    desc_func = glob['desc_func']

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

    r_desc, r_d_desc = r_desc_d_desc or desc_func.from_R(r, lat_and_inv)

    n_train = int(R_desc_perms.shape[0] / n_perms)

    wkr_start, wkr_stop = (0, n_train) if wkr_start_stop is None else wkr_start_stop
    if chunk_size is None:
        chunk_size = n_train

    dim_d, dim_i = r_d_desc.shape
    dim_c = chunk_size * n_perms

    # Pre-allocate memory.
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

        np.subtract(
            np.broadcast_to(r_desc, rj_desc_perms.shape),
            rj_desc_perms,
            out=diff_ab_perms,
        )
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

        np.exp(-norm_ab_perms / sig, out=mat52_base)
        mat52_base *= mat52_base_fact
        np.einsum(
            'ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms, out=a_x2
        )  # colum wise dot product

        F += (a_x2 * mat52_base).dot(diff_ab_perms) * diag_scale_fact
        mat52_base *= norm_ab_perms + sig
        F -= mat52_base.dot(rj_d_desc_alpha_perms)

        # Note: Energies are automatically predicted with a flipped sign here (because -E are trained, instead of E)
        E_F[0] += a_x2.dot(mat52_base)

        # Note: Energies are automatically predicted with a flipped sign here (because -E are trained, instead of E)
        if 'alphas_E_lin' in glob:

            K_fe = diff_ab_perms * mat52_base[:, None]
            F += alphas_E_lin[b_start:b_stop].dot(K_fe)

            K_ee = (
                1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            ) * np.exp(-norm_ab_perms / sig)
            E_F[0] += K_ee.dot(alphas_E_lin[b_start:b_stop])

        b_start = b_stop

    out = E_F[: dim_i + 1]

    # Descriptor has less entries than 3N, need to extend size of the 'E_F' array.
    if dim_d < dim_i:
        out = np.empty((dim_i + 1,))
        out[0] = E_F[0]

    np.dot(F, r_d_desc, out=out[1:])

    return out


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
                automatically via the function `prepare_parallel`.
                Note: Running calculations via PyTorch is only
                recommended with available GPU hardware. CPU calcuations
                are faster with our NumPy implementation.

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
        """

        global globs
        if 'globs' not in globals():
            globs = []

        # Create a personal global space for this model at a new index
        # Note: do not call delete entries in this list, since 'self.glob_id' is
        # static. Instead, setting them to None conserves positions while still
        # freeing up memory.
        globs.append({})
        self.glob_id = len(globs) - 1
        glob = globs[self.glob_id]

        self.log = logging.getLogger(__name__)

        if 'type' not in model or not (model['type'] == 'm' or model['type'] == b'm'):
            self.log.critical('The provided data structure is not a valid model.')
            sys.exit()

        self.n_atoms = model['z'].shape[0]

        self.desc = Desc(self.n_atoms, max_processes=max_processes)
        glob['desc_func'] = self.desc

        self.lat_and_inv = (
            (model['lattice'], np.linalg.inv(model['lattice']))
            if 'lattice' in model
            else None
        )

        self.n_train = model['R_desc'].shape[1]
        glob['sig'] = model['sig']

        self.std = model['std'] if 'std' in model else 1.0
        self.c = model['c']

        n_perms = model['perms'].shape[0]
        self.n_perms = n_perms
        glob['n_perms'] = n_perms

        self.tril_perms_lin = model['tril_perms_lin']

        # Precompute permuted training descriptors and its first derivatives multiplied with the coefficients (only needed for cached variant).

        R_desc_perms = (
            np.tile(model['R_desc'].T, n_perms)[:, self.tril_perms_lin]
            .reshape(self.n_train, n_perms, -1, order='F')
            .reshape(self.n_train * n_perms, -1)
        )
        glob['R_desc_perms'], glob['R_desc_perms_shape'] = share_array(R_desc_perms)

        R_d_desc_alpha_perms = (
            np.tile(model['R_d_desc_alpha'], n_perms)[:, self.tril_perms_lin]
            .reshape(self.n_train, n_perms, -1, order='F')
            .reshape(self.n_train * n_perms, -1)
        )
        glob['R_d_desc_alpha_perms'], glob['R_d_desc_alpha_perms_shape'] = share_array(
            R_d_desc_alpha_perms
        )

        if 'alphas_E' in model:
            alphas_E_lin = np.tile(model['alphas_E'][:, None], (1, n_perms)).ravel()
            glob['alphas_E_lin'], glob['alphas_E_lin_shape'] = share_array(alphas_E_lin)

        # GPU support

        self.use_torch = use_torch

        if use_torch and not _has_torch:
            raise ImportError(
                'Optional PyTorch dependency not found! Please run \'pip install sgdml[torch]\' to install it or disable the PyTorch option.'
            )

        self.torch_predict = None
        if self.use_torch:

            from .torchtools import GDMLTorchPredict

            self.torch_predict = GDMLTorchPredict(model, self.lat_and_inv)#.to(self.torch_device)

            # enable data parallelism
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                self.torch_predict = torch.nn.DataParallel(self.torch_predict)

            self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.torch_predict.to(self.torch_device)

            # is_cuda = next(self.torch_predict.parameters()).is_cuda
            # if is_cuda:
            #    self.log.info('Numbers of CUDA devices found: {:d}'.format(n_gpu))
            # else:
            #    self.log.warning(
            #        'No CUDA devices found! PyTorch is running on the CPU.'
            #    )

        # Parallel processing configuration

        self.bulk_mp = False  # Bulk predictions with multiple processes?

        # How many parallel processes?
        self.max_processes = max_processes
        if self.max_processes is None:
            self.max_processes = mp.cpu_count()
        self.pool = None
        self.num_workers = 1
        self._set_num_workers(num_workers)

        # Size of chunks in which each parallel task will be processed (unit: number of training samples)
        # This parameter should be as large as possible, but it depends on the size of available memory.
        self._set_chunk_size(batch_size)

    def __del__(self):

        global globs

        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()

        if 'globs' in globals() and globs is not None and self.glob_id < len(globs):
            globs[self.glob_id] = None


    ## Public ##

    def set_alphas(self, R_d_desc, alphas_F, alphas_E=None):
        """
        Reconfigure the current model with a new set of regression parameters.
        This is necessary when training the model iteratively.

        Parameters
        ----------
                R_d_desc : :obj:`numpy.ndarray`
                    Array containing the Jacobian of the descriptor for
                    each training point.
                alphas_F : :obj:`numpy.ndarray`
                    1D array containing the new model parameters.
                alphas_F : :obj:`numpy.ndarray`, optional
                    1D array containing the additional new model parameters, if
                    energy constraints are used in the kernel (`use_E_cstr=True`)
        """

        if self.use_torch:
            model = self.torch_predict
            if isinstance(self.torch_predict, torch.nn.DataParallel):
                model = model.module

            model.set_alphas(R_d_desc, alphas_F)

            # TODO: E_cstr does not work on GPU!

        else:

            global globs
            glob = globs[self.glob_id]

            r_dim = R_d_desc.shape[2]
            R_d_desc_alpha = np.einsum(
                'kji,ki->kj', R_d_desc, alphas_F.reshape(-1, r_dim)
            )

            R_d_desc_alpha_perms_new = np.tile(R_d_desc_alpha, self.n_perms)[
                :, self.tril_perms_lin
            ].reshape(self.n_train, self.n_perms, -1, order='F')

            R_d_desc_alpha_perms = np.frombuffer(glob['R_d_desc_alpha_perms'])
            np.copyto(R_d_desc_alpha_perms, R_d_desc_alpha_perms_new.ravel())

            if alphas_E is not None:

                alphas_E_lin_new = np.tile(alphas_E[:, None], (1, self.n_perms)).ravel()

                alphas_E_lin = np.frombuffer(glob['alphas_E_lin'])
                np.copyto(alphas_E_lin, alphas_E_lin_new)

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

        if self.num_workers is not num_workers:

            if self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None

            self.num_workers = 1
            if num_workers is None or num_workers > 1:
                self.pool = mp.Pool(processes=num_workers)
                self.num_workers = self.pool._processes

        # Data ranges for processes
        if self.bulk_mp:
            wkr_starts = [self.n_train]
        else:
            wkr_starts = list(
                range(
                    0,
                    self.n_train,
                    int(np.ceil(float(self.n_train) / self.num_workers)),
                )
            )
        wkr_stops = wkr_starts[1:] + [self.n_train]

        self.wkr_starts_stops = list(zip(wkr_starts, wkr_stops))

    def _reset_mp(self):

        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

        self.pool = mp.Pool(processes=self.num_workers)
        self.num_workers = self.pool._processes

    def _set_chunk_size(self, chunk_size=None):

        # TODO: complain if chunk or worker parameters do not fit training data (this causes issues with the caching)!!
        """
        Set chunk size for each worker process.

        Every prediction is generated as a linear combination of the training
        points that the model is comprised of. If multiple workers are available
        (and bulk mode is disabled), each one processes an (approximatelly equal)
        part of those training points. Then, the chunk size determines how much of
        a processes workload is passed to NumPy's underlying low-level routines at
        once. If the chunk size is smaller than the number of points the worker is
        supposed to process, it processes them in multiple steps using a loop. This
        can sometimes be faster, depending on the available hardware.

        Note
        ----
                This parameter can be optimally determined using
                `prepare_parallel`.

        Parameters
        ----------
                chunk_size : int
                        Chunk size (maximum value is set if `None`).
        """

        if chunk_size is None:
            chunk_size = self.n_train

        self.chunk_size = chunk_size

    def _set_batch_size(self, batch_size=None):  # deprecated
        """

        Warning
        -------
        Deprecated! Please use the function `_set_chunk_size` in future projects.

        Set chunk size for each worker process. A chunk is a subset
        of the training data points whose linear combination needs to
        be evaluated in order to generate a prediction.

        The chunk size determines how much of a processes workload will
        be passed to Python's underlying low-level routines at once.
        This parameter is highly hardware dependent.

        Note
        ----
                This parameter can be optimally determined using
                `prepare_parallel`.

        Parameters
        ----------
                batch_size : int
                        Chunk size (maximum value is set if `None`).
        """

        self._set_chunk_size(batch_size)

    def _set_bulk_mp(self, bulk_mp=False):
        """
        Toggles bulk prediction mode.

        If bulk prediction is enabled, the prediction is parallelized accross
        input geometries, i.e. each worker generates the complete prediction for
        one query. Otherwise (depending on the number of available CPU cores) the
        input geometries are process sequentially, but every one of them may be
        processed by multiple workers at once (in chunks).

        Note
        ----
                This parameter can be optimally determined using
                `prepare_parallel`.

        Parameters
        ----------
                bulk_mp : bool, optional
                        Enable or disable bulk prediction mode.
        """

        bulk_mp = bool(bulk_mp)
        if self.bulk_mp is not bulk_mp:
            self.bulk_mp = bulk_mp

            # Reset data ranges for processes stored in 'wkr_starts_stops'
            self._set_num_workers(self.num_workers)

    def set_opt_num_workers_and_batch_size_fast(self, n_bulk=1, n_reps=1):  # deprecated
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

    def prepare_parallel(
        self, n_bulk=1, n_reps=1, return_is_from_cache=False
    ):  # noqa: C901
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

        global globs
        glob = globs[self.glob_id]
        n_perms = glob['n_perms']

        # No benchmarking necessary if prediction is running on GPUs.
        if self.use_torch:
            self.log.info(
                'Skipping multi-CPU benchmark, since torch is enabled.'
            )  # TODO: clarity!
            return

        # Retrieve cached benchmark results, if available.
        bmark_result = self._load_cached_bmark_result(n_bulk)
        if bmark_result is not None:

            num_workers, chunk_size, bulk_mp, gps = bmark_result

            self._set_chunk_size(chunk_size)
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

        best_params = None

        r_dummy = np.random.rand(n_bulk, self.n_atoms * 3)

        def _dummy_predict():
            self.predict(r_dummy)

        bulk_mp_rng = [True, False] if n_bulk > 1 else [False]
        for bulk_mp in bulk_mp_rng:
            self._set_bulk_mp(bulk_mp)

            if bulk_mp is False:
                last_i = 0

            num_workers_rng = (
                list(range(self.max_processes, 1, -1))
                if bulk_mp
                else list(range(1, self.max_processes + 1))
            )

            # num_workers_rng_sizes = [batch_size for batch_size in batch_size_rng if min_batch_size % batch_size == 0]

            # for num_workers in range(min_num_workers,self.max_processes+1):
            for num_workers in num_workers_rng:
                if not bulk_mp and self.n_train % num_workers != 0:
                    continue

                self._set_num_workers(num_workers)

                best_gps = 0
                gps_rng = (np.inf, 0.0)  # min and max per num_workers

                min_chunk_size = (
                    min(self.n_train, n_bulk)
                    if bulk_mp
                    else int(np.ceil(self.n_train / num_workers))
                )
                chunk_size_rng = list(range(min_chunk_size, 0, -1))

                # for i in range(0,min_batch_size):
                chunk_size_rng_sizes = [
                    chunk_size
                    for chunk_size in chunk_size_rng
                    if min_chunk_size % chunk_size == 0
                ]

                # print('batch_size_rng_sizes ' + str(bulk_mp))
                # print(batch_size_rng_sizes)

                i_done = 0
                i_dir = 1
                i = 0 if last_i is None else last_i
                # i = 0

                # print(batch_size_rng_sizes)
                while i >= 0 and i < len(chunk_size_rng_sizes):

                    chunk_size = chunk_size_rng_sizes[i]
                    self._set_chunk_size(chunk_size)

                    i_done += 1

                    gps = n_bulk * n_reps / timeit.timeit(_dummy_predict, number=n_reps)

                    # print(
                    #   '{:2d}@{:d} {:d} | {:7.2f} gps'.format(
                    #       num_workers, chunk_size, bulk_mp, gps
                    #   )
                    # )

                    gps_rng = (
                        min(gps_rng[0], gps),
                        max(gps_rng[1], gps),
                    )  # min and max per num_workers

                    # gps_min_max = min(gps_min_max[0], gps), max(gps_min_max[1], gps)

                    # print('     best_gps ' + str(best_gps))

                    # NEW

                    # if gps > best_gps and gps > gps_min: # gps is still going up, everything is good
                    #     best_gps = gps
                    #     best_params = num_workers, batch_size, bulk_mp
                    # else:
                    #     break

                    # if gps > best_gps: # gps is still going up, everything is good
                    #     best_gps = gps
                    #     best_params = num_workers, batch_size, bulk_mp
                    # else: # gps did not go up wrt. to previous step

                    #     # can we switch the search direction?
                    #     #   did we already?
                    #     #   we checked two consecutive configurations
                    #     #   are bigger batch sizes possible?

                    #     print(batch_size_rng_sizes)

                    #     turn_search_dir = i_dir > 0 and i_done == 2 and batch_size != batch_size_rng_sizes[1]

                    #     # only turn, if the current gps is not lower than the lowest overall
                    #     if turn_search_dir and gps >= gps_min:
                    #         i -= 2 * i_dir
                    #         i_dir = -1
                    #         print('><')
                    #         continue
                    #     else:
                    #         print('>>break ' + str(i_done))
                    #         break

                    # NEW

                    # gps still going up?
                    # AND: gps not lower than the lowest overall?
                    # if gps < best_gps and gps >= gps_min:
                    if gps < best_gps:
                        if (
                            i_dir > 0
                            and i_done == 2
                            and chunk_size
                            != chunk_size_rng_sizes[
                                1
                            ]  # there is no point in turning if this is the second batch size in the range
                        ):  # do we turn?
                            i -= 2 * i_dir
                            i_dir = -1
                            # print('><')
                            continue
                        else:
                            if chunk_size == chunk_size_rng_sizes[1]:
                                i -= 1 * i_dir
                            # print('>>break ' + str(i_done))
                            break
                    else:
                        best_gps = gps
                        best_params = num_workers, chunk_size, bulk_mp

                    if (
                        not bulk_mp and n_bulk > 1
                    ):  # stop search early when multiple cpus are available and the 1 cpu case is tested
                        if (
                            gps < gps_min
                        ):  # if the batch size run is lower than the lowest overall, stop right here
                            # print('breaking here')
                            break

                    i += 1 * i_dir

                last_i = i - 1 * i_dir
                i_dir = 1

                if len(best_results) > 0:
                    overall_best_gps = max(best_results, key=lambda x: x[1])[1]
                    if best_gps < overall_best_gps:
                        # print('breaking, because best of last test was worse than overall best so far')
                        break

                    # if best_gps < gps_min:
                    #    print('breaking here3')
                    #    break

                gps_min = gps_rng[0]  # FIX me: is this the overall min?
                # print ('gps_min ' + str(gps_min))

                # print ('best_gps')
                # print (best_gps)

                best_results.append(
                    (best_params, best_gps)
                )  # best results per num_workers

        (num_workers, chunk_size, bulk_mp), gps = max(best_results, key=lambda x: x[1])

        # Cache benchmark results.
        self._save_cached_bmark_result(n_bulk, num_workers, chunk_size, bulk_mp, gps)

        self._set_chunk_size(chunk_size)
        self._set_num_workers(num_workers)
        self._set_bulk_mp(bulk_mp)

        if return_is_from_cache:
            is_from_cache = False
            return gps, is_from_cache
        else:
            return gps

    def _save_cached_bmark_result(self, n_bulk, num_workers, chunk_size, bulk_mp, gps):

        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        bmark_file = '_bmark_cache.npz'
        bmark_path = os.path.join(pkg_dir, bmark_file)

        bkey = '{}-{}-{}-{}'.format(
            self.n_atoms, self.n_train, n_bulk, self.max_processes
        )

        if os.path.exists(bmark_path):

            with np.load(bmark_path, allow_pickle=True) as bmark:
                bmark = dict(bmark)

                bmark['runs'] = np.append(bmark['runs'], bkey)
                bmark['num_workers'] = np.append(bmark['num_workers'], num_workers)
                bmark['batch_size'] = np.append(bmark['batch_size'], chunk_size)
                bmark['bulk_mp'] = np.append(bmark['bulk_mp'], bulk_mp)
                bmark['gps'] = np.append(bmark['gps'], gps)
        else:
            bmark = {
                'code_version': __version__,
                'runs': [bkey],
                'gps': [gps],
                'num_workers': [num_workers],
                'batch_size': [chunk_size],
                'bulk_mp': [bulk_mp],
            }

        np.savez_compressed(bmark_path, **bmark)

    def _load_cached_bmark_result(self, n_bulk):

        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        bmark_file = '_bmark_cache.npz'
        bmark_path = os.path.join(pkg_dir, bmark_file)

        bkey = '{}-{}-{}-{}'.format(
            self.n_atoms, self.n_train, n_bulk, self.max_processes
        )

        if not os.path.exists(bmark_path):
            return None

        with np.load(bmark_path, allow_pickle=True) as bmark:

            # Keep collecting benchmark runs, until we have at least three.
            run_idxs = np.where(bmark['runs'] == bkey)[0]
            if len(run_idxs) >= 3:

                config_keys = []
                for run_idx in run_idxs:
                    config_keys.append(
                        '{}-{}-{}'.format(
                            bmark['num_workers'][run_idx],
                            bmark['batch_size'][run_idx],
                            bmark['bulk_mp'][run_idx],
                        )
                    )

                values, uinverse = np.unique(config_keys, return_index=True)

                best_mean = -1
                best_gps = 0
                for i, config_key in enumerate(zip(values, uinverse)):
                    mean_gps = np.mean(
                        bmark['gps'][
                            np.where(np.array(config_keys) == config_key[0])[0]
                        ]
                    )

                    if best_gps == 0 or best_gps < mean_gps:
                        best_mean = i
                        best_gps = mean_gps

                best_idx = run_idxs[uinverse[best_mean]]
                num_workers = bmark['num_workers'][best_idx]
                chunk_size = bmark['batch_size'][best_idx]
                bulk_mp = bmark['bulk_mp'][best_idx]

                return num_workers, chunk_size, bulk_mp, best_gps

        return None

    def get_GPU_batch(self):
        """
        Get batch size used by the GPU implementation to process bulk
        predictions (predictions for multiple input geometries at once).

        This value is determined on-the-fly depending on the available GPU
        memory.
        """

        if self.use_torch:

            model = self.torch_predict
            if isinstance(self.torch_predict, torch.nn.DataParallel):
                model = model.module

            return model._batch_size()

    def predict(self, R, R_desc=None, R_d_desc=None):
        """
        Predict energy and forces for multiple geometries. This function
        can run on the GPU, if the optional PyTorch dependency is
        installed and `use_torch=True` was speciefied during
        initialization of this class.

        Optionally, the descriptors and descriptor Jacobians for the
        same geometries can be provided, if already available from some
        previous calculations.

        Note
        ----
                The order of the atoms in `R` is not arbitrary and must
                be the same as used for training the model.

        Parameters
        ----------
                R : :obj:`numpy.ndarray`
                        An 2D array of size M x 3N containing the
                        Cartesian coordinates of each atom of M
                        molecules.
                R_desc : :obj:`numpy.ndarray`, optional
                        An 2D array of size M x D containing the
                        descriptors of dimension D for M
                        molecules.
                R_d_desc : :obj:`numpy.ndarray`, optional
                        A 2D array of size M x D x 3N containing of the
                        descriptor Jacobians for M molecules. The descriptor
                        has dimension D with 3N partial derivatives with
                        respect to the 3N Cartesian coordinates of each atom.

        Returns
        -------
                :obj:`numpy.ndarray`
                        Energies stored in an 1D array of size M.
                :obj:`numpy.ndarray`
                        Forces stored in an 2D arry of size M x 3N.
        """

        # Use precomputed descriptors in training mode.
        train_mode = R_desc is not None and R_d_desc is not None

        # Add singleton dimension if input is (,3N).
        if R.ndim == 1:
            R = R[None, :]

        if self.use_torch:  # multi-GPU (or CPU if no GPUs are available)

            n_train = R.shape[0]
            R_torch = torch.from_numpy(R.reshape(n_train, -1, 3)).to(self.torch_device)

            E_torch, F_torch = self.torch_predict.forward(R_torch)

            E = E_torch.cpu().numpy()
            F = F_torch.cpu().numpy().reshape(n_train, -1)

        else:  # multi-CPU

            n_pred, dim_i = R.shape
            E_F = np.empty((n_pred, dim_i + 1))

            if self.bulk_mp:  # One whole prediction per worker (and multiple workers).

                _predict_wo_r_or_desc = partial(
                    _predict_wkr,
                    lat_and_inv=self.lat_and_inv,
                    glob_id=self.glob_id,
                    wkr_start_stop=None,
                    chunk_size=self.chunk_size,
                )

                for i, e_f in enumerate(
                    self.pool.imap(
                        partial(_predict_wo_r_or_desc, None)
                        if train_mode
                        else partial(_predict_wo_r_or_desc, r_desc_d_desc=None),
                        zip(R_desc, R_d_desc) if train_mode else R,
                    )
                ):
                    E_F[i, :] = e_f

            else:  # Multiple workers per prediction (or just one worker).

                for i, r in enumerate(R):

                    if train_mode:
                        r_desc, r_d_desc = R_desc[i], R_d_desc[i]
                    else:
                        r_desc, r_d_desc = self.desc.from_R(r, self.lat_and_inv)

                    _predict_wo_wkr_starts_stops = partial(
                        _predict_wkr,
                        None,
                        (r_desc, r_d_desc),
                        self.lat_and_inv,
                        self.glob_id,
                        chunk_size=self.chunk_size,
                    )

                    if self.num_workers == 1:
                        E_F[i, :] = _predict_wo_wkr_starts_stops()
                    else:
                        E_F[i, :] = sum(
                            self.pool.imap_unordered(
                                _predict_wo_wkr_starts_stops, self.wkr_starts_stops
                            )
                        )

            E_F *= self.std
            F = E_F[:, 1:]
            E = E_F[:, 0] + self.c

        return E, F

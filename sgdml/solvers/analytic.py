#!/usr/bin/python

# MIT License
#
# Copyright (c) 2020-2022 Stefan Chmiela
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
import warnings
from functools import partial

import numpy as np
import scipy as sp
import timeit

from .. import DONE, NOT_DONE


class Analytic(object):
    def __init__(self, gdml_train, desc, callback=None):

        self.log = logging.getLogger(__name__)

        self.gdml_train = gdml_train
        self.desc = desc

        self.callback = callback

    # from memory_profiler import profile
    # @profile
    def solve(self, task, R_desc, R_d_desc, tril_perms_lin, y):

        sig = task['sig']
        lam = task['lam']
        use_E_cstr = task['use_E_cstr']

        n_train, dim_d = R_d_desc.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms

        if self.callback is not None:
            self.callback = partial(
                self.callback,
                disp_str='Assembling kernel matrix',
            )

        K = -self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            callback=self.callback,
        )  # Flip sign to make convex

        start = timeit.default_timer()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if K.shape[0] == K.shape[1]:

                K[np.diag_indices_from(K)] += lam  # Regularize

                if self.callback is not None:
                    self.callback = partial(
                        self.callback,
                        disp_str='Solving linear system (Cholesky factorization)',
                    )
                    self.callback(NOT_DONE)

                try:

                    # Cholesky (do not overwrite K in case we need to retry)
                    L, lower = sp.linalg.cho_factor(
                        K, overwrite_a=False, check_finite=False
                    )
                    alphas = -sp.linalg.cho_solve(
                        (L, lower), y, overwrite_b=False, check_finite=False
                    )

                except np.linalg.LinAlgError:  # Try a solver that makes less assumptions

                    if self.callback is not None:
                        self.callback = partial(
                            self.callback,
                            disp_str='Solving linear system (LU factorization)      ',  # Keep whitespaces!
                        )
                        self.callback(NOT_DONE)

                    try:
                        # LU
                        alphas = -sp.linalg.solve(
                            K, y, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    except MemoryError:
                        self.log.critical(
                            'Not enough memory to train this system using a closed form solver.'
                        )
                        print()
                        os._exit(1)

                except MemoryError:
                    self.log.critical(
                        'Not enough memory to train this system using a closed form solver.'
                    )
                    print()
                    os._exit(1)
            else:

                if self.callback is not None:
                    self.callback = partial(
                        self.callback,
                        disp_str='Solving over-determined linear system (least squares approximation)',
                    )
                    self.callback(NOT_DONE)

                # Least squares for non-square K
                alphas = -np.linalg.lstsq(K, y, rcond=-1)[0]

        stop = timeit.default_timer()

        if self.callback is not None:
            dur_s = stop - start
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            self.callback(
                DONE,
                disp_str='Training on {:,} points'.format(n_train),
                sec_disp_str=sec_disp_str,
            )

        return alphas

    @staticmethod
    def est_memory_requirement(n_train, n_atoms):

        est_bytes = 3 * (n_train * 3 * n_atoms) ** 2 * 8  # K + factor(s) of K
        est_bytes += (n_train * 3 * n_atoms) * 8  # alpha

        return est_bytes

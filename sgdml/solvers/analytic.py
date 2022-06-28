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

    #from memory_profiler import profile
    #@profile
    def solve(self, task, R_desc, R_d_desc, tril_perms_lin, y):

        sig = task['sig']
        lam = task['lam']
        use_E_cstr = task['use_E_cstr']

        n_train, dim_d = R_d_desc.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms

        # Compress kernel based on symmetries
        col_idxs = np.s_[:]
        if 'cprsn_keep_atoms_idxs' in task:

            cprsn_keep_idxs = task['cprsn_keep_atoms_idxs']
            cprsn_keep_idxs_lin = (
                np.arange(dim_i).reshape(n_atoms, -1)[cprsn_keep_idxs, :].ravel()
            )

            # if cprsn_callback is not None:
            #    cprsn_callback(n_atoms, cprsn_keep_idxs.shape[0])

            col_idxs = (
                cprsn_keep_idxs_lin[:, None] + np.arange(n_train) * dim_i
            ).T.ravel()

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
            col_idxs=col_idxs,
            callback=self.callback,
        ) # flip sign to make convex





        #import matplotlib.pyplot as plt

        #plt.imshow(K, cmap='PiYG', interpolation='nearest')
        #plt.show()

        #print('analytic!')

        #sys.exit()





        start = timeit.default_timer()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if K.shape[0] == K.shape[1]:

                K[np.diag_indices_from(K)] += lam  # regularize


                # def block_view(A, block=(3, 3)):
                #     """Provide a 2D block view to 2D array. No error checking made.
                #     Therefore meaningful (as implemented) only for blocks strictly
                #     compatible with the shape of A."""
                #     # simple shape and strides computations may seem at first strange
                #     # unless one is able to recognize the 'tuple additions' involved ;-)

                #     shape = (A.shape[0] // block[0], A.shape[1] // block[1]) + block
                #     strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
                #     return np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)


                # # NEW

                # K_stripped = np.zeros(K.shape)

                # K_blocked = block_view(K)


                # #k_atom = np.zeros((n_train*3, n_train*3))
                # #y_atom = np.zeros((n_train*3,))
               

                # K_stripped_blocked = block_view(K_stripped)



                # for a in range(n_atoms):
                #     K_stripped_blocked[a::n_atoms, a::n_atoms] = K_blocked[a::n_atoms, a::n_atoms]

                # #for a in range(n_atoms):
                # #    for i in range(3):
                #         #k_atom[i::3, i::3] = K[i::(n_atoms*3), i::(n_atoms*3)]
                #         #y_atom[i::3] = y[i::(n_atoms*3)]



                #         #for j in range(3):
                #         #    K_stripped[(a*3+i)::(n_atoms*3), (a*3+j)] = K[(a*3+i)::(n_atoms*3), (a*3+j)]


                # #import matplotlib.pyplot as plt

                # #plt.imshow(np.abs(K_stripped), cmap='PiYG')
                # #plt.colorbar()
                # #plt.show()

                # K = K_stripped

                # #a_atom = np.linalg.solve(-k_atom, y_atom)

                # L, lower = sp.linalg.cho_factor(
                #     -k_atom, overwrite_a=True, check_finite=False
                # )
                # a_atom = -sp.linalg.cho_solve(
                #     (L, lower), y_atom, overwrite_b=True, check_finite=False
                # )

                # print(a_atom)

                #Kop = K.copy()
                #yop = y.copy()

                # NEW

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

                except np.linalg.LinAlgError:  # try a solver that makes less assumptions

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

                # least squares for non-square K
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

        est_bytes = 3 * (n_train * 3 * n_atoms) ** 2 * 8 # K + factor(s) of K
        est_bytes += (n_train * 3 * n_atoms) * 8 # alpha

        return est_bytes

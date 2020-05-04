#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2020 Stefan Chmiela, Luis Galvez
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
import scipy as sp
import multiprocessing as mp
from functools import partial
from scipy import spatial
import timeit

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True


def _from_r_alias(obj, r, lat_and_inv=None):
    """
    Alias for instance method that allows the method to be called in a 
    multiprocessing pool
    """
    return obj._from_r(r, lat_and_inv=lat_and_inv)


class Desc(object):
    def __init__(self, n_atoms, max_processes=None):
        """
        Generate descriptors and their Jacobians for molecular geometries,
        including support for periodic boundary conditions.

        Parameters
        ----------
                n_atoms : int
                        Number of atoms in the represented system.
                max_processes : int, optional
                        Limit the max. number of processes. Otherwise
                        all CPU cores are used. This parameters has no
                        effect if `use_torch=True`.
        """

        self.n_atoms = n_atoms
        self.dim_i = 3 * n_atoms

        # Size of the resulting descriptor vector.
        self.dim = (n_atoms * (n_atoms - 1)) // 2

        # Precompute indices for nonzero entries in desriptor derivatives.
        self.d_desc_mask = np.zeros((n_atoms, n_atoms - 1), dtype=np.int)
        for a in range(n_atoms):  # for each partial derivative
            rows, cols = np.tril_indices(n_atoms, -1)
            self.d_desc_mask[a, :] = np.concatenate(
                [np.where(rows == a)[0], np.where(cols == a)[0]]
            )

        self.M = np.arange(1, n_atoms)  # indexes matrix row-wise, skipping diagonal
        for a in range(1, n_atoms):
            self.M = np.concatenate((self.M, np.delete(np.arange(n_atoms), a)))

        self.A = np.repeat(
            np.arange(n_atoms), n_atoms - 1
        )  # [0, 0, ..., 1, 1, ..., 2, 2, ...]

        self.d_desc = np.zeros(
            (self.dim, n_atoms, 3)
        )  # template for descriptor matrix (zeros are important)

        self.max_processes = max_processes

    def from_R(self, R, lat_and_inv=None, callback=None):
        """
        Generate descriptor and its Jacobian for multiple molecular geometries
        in Cartesian coordinates.

        Parameters
        ----------
            R : :obj:`numpy.ndarray`
                Array of size M x 3N containing the Cartesian coordinates of
                each atom.
            lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
                Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
            callback : callable, optional
                Descriptor and descriptor Jacobian generation status.
                    current : int
                        Current progress (number of completed descriptors).
                    total : int
                        Task size (total number of descriptors to create).
                    sec_disp_str : :obj:`str`, optional
                        Once complete, this string contains the
                        time it took complete this task (seconds).

        Returns
        -------
            :obj:`numpy.ndarray`
                Array of size M x N(N-1)/2 containing the descriptor representation
                for each geometry.
            :obj:`numpy.ndarray`
                Array of size M x N(N-1)/2 x 3N containing all partial
                derivatives of the descriptor for each geometry.
        """

        # Add singleton dimension if input is (,3N).
        if R.ndim == 1:
            R = R[None, :]

        M = R.shape[0]
        if M == 1:
            return self._from_r(R, lat_and_inv)

        R_desc = np.empty([M, self.dim])
        R_d_desc = np.empty([M, self.dim, self.dim_i])

        # Generate descriptor and their Jacobians
        start = timeit.default_timer()
        pool = mp.Pool(self.max_processes)
        for i, r_desc_r_d_desc in enumerate(
            pool.imap(partial(_from_r_alias, self, lat_and_inv=lat_and_inv), R)
        ):
            R_desc[i, :], R_d_desc[i, :, :] = r_desc_r_d_desc

            if callback is not None and i < M - 1:
                callback(i, M - 1)

        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        stop = timeit.default_timer()

        if callback is not None:
            dur_s = (stop - start) / 2
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            callback(M, M, sec_disp_str=sec_disp_str)

        return R_desc, R_d_desc

    def pbc_diff(self, diffs, lat_and_inv, use_torch=False):
        """
        Clamp differences of vectors to super cell.

        Parameters
        ----------
            diffs : :obj:`numpy.ndarray`
                N x 3 matrix of N pairwise differences between vectors `u - v`
            lat_and_inv : tuple of :obj:`numpy.ndarray`
                Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
            use_torch : boolean, optional
                Enable, if the inputs are PyTorch objects.

        Returns
        -------
            :obj:`numpy.ndarray`
                N x 3 matrix clamped differences
        """

        lat, lat_inv = lat_and_inv

        if use_torch and not _has_torch:
            raise ImportError(
                'Optional PyTorch dependency not found! Please run \'pip install sgdml[torch]\' to install it or disable the PyTorch option.'
            )

        if use_torch:
            c = lat_inv.mm(diffs.t())
            diffs -= lat.mm(c.round()).t()
        else:
            c = lat_inv.dot(diffs.T)
            diffs -= lat.dot(np.rint(c)).T

        return diffs

    def perm(self, perm):
        """
        Convert atom permutation to descriptor permutation.

        A permutation of N atoms is converted to a permutation that acts on
        the corresponding descriptor representation. Applying the converted
        permutation to a descriptor is equivalent to permuting the atoms 
        first and then generating the descriptor. 

        Parameters
        ----------
            perm : :obj:`numpy.ndarray`
                Array of size N containing the atom permutation.

        Returns
        -------
            :obj:`numpy.ndarray`
                Array of size N(N-1)/2 containing the corresponding
                descriptor permutation.
        """

        n = len(perm)

        rest = np.zeros((n, n))
        rest[np.tril_indices(n, -1)] = list(range((n ** 2 - n) // 2))
        rest = rest + rest.T
        rest = rest[perm, :]
        rest = rest[:, perm]

        return rest[np.tril_indices(n, -1)].astype(int)

    # Private

    def _from_r(self, r, lat_and_inv=None):
        """
        Generate descriptor and its Jacobian for one molecular geometry
        in Cartesian coordinates.

        Parameters
        ----------
            r : :obj:`numpy.ndarray`
                Array of size 3N containing the Cartesian coordinates of
                each atom.
            lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
                Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.

        Returns
        -------
            :obj:`numpy.ndarray`
                Descriptor representation as 1D array of size N(N-1)/2
            :obj:`numpy.ndarray`
                Array of size N(N-1)/2 x 3N containing all partial
                derivatives of the descriptor.
        """

        # Add singleton dimension if input is (,3N).
        if r.ndim == 1:
            r = r[None, :]

        pd = self._pdist(r, lat_and_inv)

        r_desc = self._r_to_desc(r, pd)
        r_d_desc = self._r_to_d_desc(r, pd, lat_and_inv)

        return r_desc, r_d_desc

    def _pdist(self, r, lat_and_inv=None):
        """
        Compute pairwise Euclidean distance matrix between all atoms.

        Parameters
        ----------
            r : :obj:`numpy.ndarray`
                Array of size 3N containing the Cartesian coordinates of
                each atom.
            lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
                Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.

        Returns
        -------
            :obj:`numpy.ndarray`
                Array of size N x N containing all pairwise distances between atoms.
        """

        r = r.reshape(-1, 3)

        if lat_and_inv is None:
            pdist = sp.spatial.distance.pdist(r, 'euclidean')
        else:
            pdist = sp.spatial.distance.pdist(
                r, lambda u, v: np.linalg.norm(self.pbc_diff(u - v, lat_and_inv))
            )

        return sp.spatial.distance.squareform(pdist, checks=False)

    def _r_to_desc(self, r, pdist):
        """
        Generate descriptor for a set of atom positions in Cartesian
        coordinates.

        Parameters
        ----------
            r : :obj:`numpy.ndarray`
                Array of size 3N containing the Cartesian coordinates of
                each atom.
            pdist : :obj:`numpy.ndarray`
                Array of size N x N containing the Euclidean distance
                (2-norm) for each pair of atoms.

        Returns
        -------
            :obj:`numpy.ndarray`
                Descriptor representation as 1D array of size N(N-1)/2
        """

        # Add singleton dimension if input is (,3N).
        if r.ndim == 1:
            r = r[None, :]

        return 1.0 / pdist[np.tril_indices(self.n_atoms, -1)]

    def _r_to_d_desc(self, r, pdist, lat_and_inv=None):
        """
        Generate descriptor Jacobian for a set of atom positions in
        Cartesian coordinates.
        This method can apply the minimum-image convention as periodic
        boundary condition for distances between atoms, given the edge
        length of the (square) unit cell.

        Parameters
        ----------
            r : :obj:`numpy.ndarray`
                Array of size 3N containing the Cartesian coordinates of
                each atom.
            pdist : :obj:`numpy.ndarray`
                Array of size N x N containing the Euclidean distance
                (2-norm) for each pair of atoms.
            lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
                Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.

        Returns
        -------
            :obj:`numpy.ndarray`
                Array of size N(N-1)/2 x 3N containing all partial
                derivatives of the descriptor.
        """

        r = r.reshape(-1, 3)

        np.seterr(divide='ignore', invalid='ignore')

        pdiff = r[:, None] - r[None, :]  # pairwise differences ri - rj
        if lat_and_inv is not None:
            pdiff = self.pbc_diff(
                pdiff.reshape(self.n_atoms ** 2, 3), lat_and_inv
            ).reshape(self.n_atoms, self.n_atoms, 3)

        d_desc_elem = pdiff / (pdist ** 3)[:, :, None]
        self.d_desc[self.d_desc_mask.ravel(), self.A, :] = d_desc_elem[
            self.M, self.A, :
        ]

        return self.d_desc.reshape(self.dim, self.dim_i)

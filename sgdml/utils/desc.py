#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2022 Stefan Chmiela, Luis Galvez
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
from scipy import spatial

import multiprocessing as mp

Pool = mp.get_context('fork').Pool

from functools import partial
import timeit

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True


def _pbc_diff(diffs, lat_and_inv, use_torch=False):
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
        diffs -= lat.dot(np.around(c)).T

    return diffs


def _pdist(r, lat_and_inv=None):
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
            Array of size N(N-1)/2 containing the upper triangle of the pairwise
            distance matrix between atoms.
    """

    r = r.reshape(-1, 3)
    n_atoms = r.shape[0]

    if lat_and_inv is None:
        pdist = sp.spatial.distance.pdist(r, 'euclidean')
    else:
        pdist = sp.spatial.distance.pdist(
            r, lambda u, v: np.linalg.norm(_pbc_diff(u - v, lat_and_inv))
        )

    tril_idxs = np.tril_indices(n_atoms, k=-1)
    return sp.spatial.distance.squareform(pdist, checks=False)[tril_idxs]


def _squareform(vec_or_mat):

    # vector to matrix representation
    if vec_or_mat.ndim == 1:

        n_tril = vec_or_mat.size
        n = int((1 + np.sqrt(8 * n_tril + 1)) / 2)

        i, j = np.tril_indices(n, k=-1)

        mat = np.zeros((n, n))
        mat[i, j] = vec_or_mat
        mat[j, i] = vec_or_mat

        return mat

    else:  # matrix to vector

        assert vec_or_mat.shape[0] == vec_or_mat.shape[1]  # matrix is square

        n = vec_or_mat.shape[0]
        i, j = np.tril_indices(n, k=-1)

        return vec_or_mat[i, j]


def _r_to_desc(r, pdist):
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

    return 1.0 / pdist


def _r_to_d_desc(r, pdist, lat_and_inv=None):
    """
    Generate descriptor Jacobian for a set of atom positions in
    Cartesian coordinates.

    This method can apply the minimum-image convention as periodic
    boundary condition for distances between atoms, given the lattice vectors.

    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 3N containing the Cartesian coordinates of
            each atom.
        pdist : :obj:`numpy.ndarray`
            Array of size N x N containing the Euclidean distance
            (2-norm) for each pair of atoms.
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.

    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N(N-1)/2 x 3N containing all partial
            derivatives of the descriptor.
    """

    r = r.reshape(-1, 3)
    pdiff = r[:, None] - r[None, :]  # pairwise differences ri - rj

    n_atoms = r.shape[0]
    i, j = np.tril_indices(n_atoms, k=-1)

    pdiff = pdiff[i, j, :]  # lower triangular

    if lat_and_inv is not None:
        pdiff = _pbc_diff(pdiff, lat_and_inv)

    d_desc_elem = pdiff / (pdist ** 3)[:, None]

    return d_desc_elem


def _from_r(r, lat_and_inv=None):
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

    pd = _pdist(r, lat_and_inv)

    r_desc = _r_to_desc(r, pd)
    r_d_desc = _r_to_d_desc(r, pd, lat_and_inv)

    return r_desc, r_d_desc


class Desc(object):
    # def __init__(self, n_atoms, interact_cut_off=None, max_processes=None):
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
                        all CPU cores are used.
        """

        self.n_atoms = n_atoms
        self.dim_i = 3 * n_atoms

        # Size of the resulting descriptor vector.
        self.dim = (n_atoms * (n_atoms - 1)) // 2

        self.tril_indices = np.tril_indices(n_atoms, k=-1)

        # Precompute indices for nonzero entries in desriptor derivatives.
        self.d_desc_mask = np.zeros((n_atoms, n_atoms - 1), dtype=np.int)
        for a in range(n_atoms):  # for each partial derivative
            rows, cols = self.tril_indices
            self.d_desc_mask[a, :] = np.concatenate(
                [np.where(rows == a)[0], np.where(cols == a)[0]]
            )

        self.dim_range = np.arange(self.dim)  # [0, 1, ..., dim-1]

        # Precompute indices for nonzero entries in desriptor derivatives.

        self.M = np.arange(1, n_atoms)  # indexes matrix row-wise, skipping diagonal
        for a in range(1, n_atoms):
            self.M = np.concatenate((self.M, np.delete(np.arange(n_atoms), a)))

        self.A = np.repeat(
            np.arange(n_atoms), n_atoms - 1
        )  # [0, 0, ..., 1, 1, ..., 2, 2, ...]

        self.max_processes = max_processes

    def from_R(self, R, lat_and_inv=None, max_processes=None, callback=None):
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
            max_processes : int, optional
                Limit the max. number of processes. Otherwise
                all CPU cores are used. This parameter overwrites the global setting as
                set during initialization.
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
            return _from_r(R, lat_and_inv)

        R_desc = np.empty([M, self.dim])
        R_d_desc = np.empty([M, self.dim, 3])

        # Generate descriptor and their Jacobians
        start = timeit.default_timer()

        pool = None
        map_func = map
        max_processes = max_processes or self.max_processes
        if max_processes != 1 and mp.cpu_count() > 1:
            pool = Pool((max_processes or mp.cpu_count()) - 1)  # exclude main process
            map_func = pool.imap

        for i, r_desc_r_d_desc in enumerate(
            map_func(partial(_from_r, lat_and_inv=lat_and_inv), R)
        ):
            R_desc[i, :], R_d_desc[i, :, :] = r_desc_r_d_desc

            if callback is not None and i < M - 1:
                callback(i, M - 1)

        if pool is not None:
            pool.close()
            pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
            pool = None

        stop = timeit.default_timer()

        if callback is not None:
            dur_s = stop - start
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            callback(M, M, sec_disp_str=sec_disp_str)

        return R_desc, R_d_desc

    # Multiplies descriptor(s) jacobian with 3N-vector(s) from the right side
    def d_desc_dot_vec(self, R_d_desc, vecs, overwrite_vecs=False):

        if R_d_desc.ndim == 2:
            R_d_desc = R_d_desc[None, ...]

        if vecs.ndim == 1:
            vecs = vecs[None, ...]

        i, j = self.tril_indices

        vecs = vecs.reshape(vecs.shape[0], -1, 3)

        einsum = np.einsum
        if _has_torch and torch.is_tensor(R_d_desc):
            assert torch.is_tensor(vecs)
            einsum = torch.einsum

        return einsum('...ij,...ij->...i', R_d_desc, vecs[:, j, :] - vecs[:, i, :])

    # Multiplies descriptor(s) jacobian with N(N-1)/2-vector(s) from the left side
    def vec_dot_d_desc(self, R_d_desc, vecs, out=None):

        if R_d_desc.ndim == 2:
            R_d_desc = R_d_desc[None, ...]

        if vecs.ndim == 1:
            vecs = vecs[None, ...]

        assert (
            R_d_desc.shape[0] == 1
            or vecs.shape[0] == 1
            or R_d_desc.shape[0] == vecs.shape[0]
        )  # either multiple descriptors or multiple vectors at once, not both (or the same number of both, than it will must be a multidot)

        n = np.max((R_d_desc.shape[0], vecs.shape[0]))
        i, j = self.tril_indices

        out = np.zeros((n, self.n_atoms, self.n_atoms, 3))
        out[:, i, j, :] = R_d_desc * vecs[..., None]
        out[:, j, i, :] = -out[:, i, j, :]
        return out.sum(axis=1).reshape(n, -1)

        # if out is None or out.shape != (n, self.n_atoms*3):
        #    out = np.zeros((n, self.n_atoms*3))

        # R_d_desc_full = np.zeros((self.n_atoms, self.n_atoms, 3))
        # for a in range(n):

        #   R_d_desc_full[i, j, :] = R_d_desc * vecs[a, :, None]
        #    R_d_desc_full[j, i, :] = -R_d_desc_full[i, j, :]
        #    out[a,:] = R_d_desc_full.sum(axis=0).ravel()

        # return out

    def d_desc_from_comp(self, R_d_desc, out=None):
        """
        Convert a compressed representation of a descriptor Jacobian back
        to its full representation.

        The compressed representation omits all zeros and scales with N
        instead of N(N-1)/2.

        Parameters
        ----------
            R_d_desc : :obj:`numpy.ndarray` or :obj:`torch.tensor`
                Array of size M x N x N x 3 containing the compressed
                descriptor Jacobian.
            out : :obj:`numpy.ndarray` or :obj:`torch.tensor`, optional
                Output argument. This must have the exact kind that would
                be returned if it was not used.

        Note
        ----
                If used, the output argument must be initialized with zeros!

        Returns
        -------
            :obj:`numpy.ndarray` or :obj:`torch.tensor`
                Array of size M x N(N-1)/2 x 3N containing the full
                representation.
        """

        if R_d_desc.ndim == 2:
            R_d_desc = R_d_desc[None, ...]

        n = R_d_desc.shape[0]
        i, j = self.tril_indices

        if out is None:
            if _has_torch and torch.is_tensor(R_d_desc):
                device = R_d_desc.device
                dtype = R_d_desc.dtype
                out = torch.zeros((n, self.dim, self.n_atoms, 3), device=device).to(
                    dtype
                )
            else:
                out = np.zeros((n, self.dim, self.n_atoms, 3))
        else:
            out = out.reshape(n, self.dim, self.n_atoms, 3)

        out[:, self.dim_range, j, :] = R_d_desc
        out[:, self.dim_range, i, :] = -R_d_desc

        return out.reshape(-1, self.dim, self.dim_i)

    def d_desc_to_comp(self, R_d_desc):
        """
        Convert a descriptor Jacobian to a compressed representation.

        The compressed representation omits all zeros and scales with N
        instead of N(N-1)/2.

        Parameters
        ----------
            R_d_desc : :obj:`numpy.ndarray`
                Array of size M x N(N-1)/2 x 3N containing the descriptor
                Jacobian.

        Returns
        -------
            :obj:`numpy.ndarray`
                Array of size M x N x N x 3 containing the compressed
                representation.
        """

        # Add singleton dimension for single inputs.
        if R_d_desc.ndim == 2:
            R_d_desc = R_d_desc[None, ...]

        n = R_d_desc.shape[0]
        n_atoms = int(R_d_desc.shape[2] / 3)

        R_d_desc = R_d_desc.reshape(n, -1, n_atoms, 3)

        ret = np.zeros((n, n_atoms, n_atoms, 3))
        ret[:, self.M, self.A, :] = R_d_desc[:, self.d_desc_mask.ravel(), self.A, :]

        # Take the upper triangle.
        i, j = self.tril_indices
        return ret[:, i, j, :]

    @staticmethod
    def perm(perm):
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

#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2019 Stefan Chmiela, Luis Galvez
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

d_desc_mask = None


def init(n_atoms):
    global d_dim, d_desc_mask, M, A, d_desc

    # Descriptor space dimension.
    d_dim = (n_atoms * (n_atoms - 1)) // 2

    # Precompute indices for nonzero entries in desriptor derivatives.
    d_desc_mask = np.zeros((n_atoms, n_atoms - 1), dtype=np.int)
    for a in range(n_atoms):  # for each partial derivative
        rows, cols = np.tril_indices(n_atoms, -1)
        d_desc_mask[a, :] = np.concatenate(
            [np.where(rows == a)[0], np.where(cols == a)[0]]
        )

    M = np.arange(1, n_atoms)  # indexes matrix row-wise, skipping diagonal
    for a in range(1, n_atoms):
        M = np.concatenate((M, np.delete(np.arange(n_atoms), a)))

    # A = (np.ones((n_atoms,n_atoms-1), int) * np.arange(n_atoms)[:,None]).ravel() # [0, 0, ..., 1, 1, ..., 2, 2, ...]
    A = np.repeat(np.arange(n_atoms), n_atoms - 1)  # [0, 0, ..., 1, 1, ..., 2, 2, ...]

    d_desc = np.zeros(
        (d_dim, n_atoms, 3)
    )  # template for descriptor matrix (zeros are important)


def pbc_diff(diffs, lat_and_inv):  # diffs: -> N x 3 matrix
    """
    Clamp differences of vectors to super cell.

    Parameters
    ----------
        diffs : :obj:`numpy.ndarray`
            N x 3 matrix of N pairwise differences between vectors `u - v`
        v : :obj:`numpy.ndarray`
            Second vector.
        lat_and_inv : tuple of :obj:`numpy.ndarray`
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.

    Returns
    -------
        :obj:`numpy.ndarray`
            N x 3 matrix clamped differences
    """

    lat, lat_inv = lat_and_inv

    c = lat_inv.dot(diffs.T)
    diffs -= lat.dot(np.rint(c)).T

    return diffs


def pbc_diff_torch(diffs, lat_and_inv):  # diffs: -> N x 3 matrix
    """
    Clamp differences of vectors to super cell (for torch tensors).

    Parameters
    ----------
        diffs : :obj:`numpy.ndarray`
            N x 3 matrix of N pairwise differences between vectors `u - v`
        v : :obj:`numpy.ndarray`
            Second vector.
        lat_and_inv : tuple of :obj:`numpy.ndarray`
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.

    Returns
    -------
        :obj:`numpy.ndarray`
            N x 3 matrix clamped differences
    """

    import torch

    lat, lat_inv = lat_and_inv

    c = lat_inv.mm(diffs.t())
    diffs -= lat.mm(c.round()).t()

    return diffs


def pdist(r, lat_and_inv=None):  # r: -> N x 3 matrix

    if lat_and_inv is None:
        pdist = sp.spatial.distance.pdist(r, 'euclidean')
    else:
        pdist = sp.spatial.distance.pdist(
            r, lambda u, v: np.linalg.norm(pbc_diff(u - v, lat_and_inv))
        )

    return sp.spatial.distance.squareform(pdist, checks=False)


def r_to_desc(r, pdist):
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

    n_atoms = r.shape[0]
    return 1.0 / pdist[np.tril_indices(n_atoms, -1)]


def r_to_d_desc(r, pdist, lat_and_inv=None):
    """
    Generate descriptor Jacobian for a set of atom positions in
    Cartesian coordinates.
    This method can apply the minimum-image convention as periodic
    boundary condition for distances between atoms, given the edge
    length of the (square) unit cell.

    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 1 x 3N containing the Cartesian coordinates of
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

    global d_dim, d_desc_mask, M, A, d_desc

    n_atoms = r.shape[0]
    if d_desc_mask is None or d_desc_mask.shape[0] != n_atoms:
        init(n_atoms)

    np.seterr(divide='ignore', invalid='ignore')

    pdiff = r[:, None] - r[None, :]  # pairwise differences ri - rj
    if lat_and_inv is not None:
        pdiff = pbc_diff(pdiff.reshape(n_atoms ** 2, 3), lat_and_inv).reshape(
            n_atoms, n_atoms, 3
        )

    d_desc_elem = pdiff / (pdist ** 3)[:, :, None]
    d_desc[d_desc_mask.ravel(), A, :] = d_desc_elem[M, A, :]

    return d_desc.reshape(d_dim, 3 * n_atoms)


def from_r(r, lat_and_inv=None):
    """
    Generate descriptor and its Jacobian for a molecular geometry
    in Cartesian coordinates.

    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 1 x 3N containing the Cartesian coordinates of
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

    pd = pdist(r, lat_and_inv)

    r_desc = r_to_desc(r, pd)
    r_d_desc = r_to_d_desc(r, pd, lat_and_inv)

    return r_desc, r_d_desc


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

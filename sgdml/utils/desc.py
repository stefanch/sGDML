import numpy as np

d_desc_mask = None


def init(n_atoms):
    global d_dim, d_desc_mask

    # Descriptor space dimension.
    d_dim = (n_atoms * (n_atoms - 1)) // 2

    # Precompute indices for nonzero entries in desriptor derivatives.
    d_desc_mask = np.zeros((n_atoms, n_atoms - 1), dtype=np.int)
    for a in range(n_atoms):  # for each partial derivative
        rows, cols = np.tril_indices(n_atoms, -1)
        d_desc_mask[a, :] = np.concatenate(
            [np.where(rows == a)[0], np.where(cols == a)[0]]
        )


# Difference with periodic boundary conditions
# b_size = 9.91241 # box size (lattice)


def pbc_diff(u, v, size):
    """
    Compute the difference of two vectors, while appling the
    minimum-image convention as periodic boundary condition.

    Parameters
    ----------
        u : :obj:`numpy.ndarray`
            First vector.
        v : :obj:`numpy.ndarray`
            Second vector.
        size : float
            Edge length of (cubic) unit cell.

    Returns
    -------
        :obj:`numpy.ndarray`
            Difference between two vectors `u - v`.
    """

    diff = u - v
    diff -= size * np.rint(diff / size)

    return diff


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


def r_to_d_desc(r, pdist, ucell_size=None):
    """
    Generate Jacobian of descriptor for a set of atom positions in
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
        ucell_size : float, optional
            Edge length of the (cubic) unit cell.

    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N(N-1)/2 x 3N containing all partial
            derivatives of the descriptor.
    """

    global d_dim, d_desc_mask

    n_atoms = r.shape[0]

    if d_desc_mask is None or d_desc_mask.shape[0] != n_atoms:
        init(n_atoms)

    np.seterr(divide='ignore', invalid='ignore')  # ignore division by zero below
    grad = np.zeros((d_dim, 3 * n_atoms))
    for a in range(n_atoms):

        if ucell_size is None:
            d_dist = (r - r[a, :]) / (pdist[a, :] ** 3)[:, None]
        else:
            d_dist = pbc_diff(r, r[a, :], ucell_size) / (pdist[a, :] ** 3)[:, None]

        idx = d_desc_mask[a, :]
        grad[idx, (3 * a) : (3 * a + 3)] = np.delete(d_dist, a, axis=0)

    return grad


def r_to_d_desc_op(r, pdist, F_d, ucell_size=None):
    """
    Compute vector-matrix product with descriptor Jacobian.

    The descriptor Jacobian will be generated and directly applied
    without storing it.
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
        F_d : :obj:`numpy.ndarray`
            Array of size N(N-1)/2.
        ucell_size : float, optional
            Edge length of the (cubic) unit cell.

    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size 3N containing the dot product of `F_d` and the
            descriptor Jacobian.
    """

    global d_dim, d_desc_mask

    n_atoms = r.shape[0]

    if d_desc_mask is None or d_desc_mask.shape[0] != n_atoms:
        init(n_atoms)

    np.seterr(divide='ignore', invalid='ignore')  # ignore division by zero below
    F_i = np.empty((3 * n_atoms,))
    for a in range(n_atoms):

        if ucell_size is None:
            d_dist = (r - r[a, :]) / (pdist[a, :] ** 3)[:, None]
        else:
            d_dist = pbc_diff(r, r[a, :], ucell_size) / (pdist[a, :] ** 3)[:, None]

        idx = d_desc_mask[a, :]
        F_d[idx].dot(np.delete(d_dist, a, axis=0), out=F_i[(3 * a) : (3 * a + 3)])

    return F_i


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

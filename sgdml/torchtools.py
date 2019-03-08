import numpy as np
import torch
import torch.nn as nn


def expand_tril(xs):
    n = int((1 + np.sqrt(8 * xs.shape[-1] + 1)) / 2)
    xs_full = torch.zeros((xs.shape[0], n, n), dtype=xs.dtype, device=xs.device)
    i, j = np.tril_indices(n, k=-1)
    xs_full[:, i, j] = xs
    xs_full[:, j, i] = xs
    i, j = np.diag_indices(n)
    xs_full[:, i, j] = 0
    return xs_full


class GDMLTorchPredict(nn.Module):
    """
    PyTorch version of :class:`~predict.GDMLPredict`. Derives from
    :class:`torch.nn.Module`. Contains no trainable parameters.
    """

    def __init__(self, model):
        super().__init__()
        self._sig = int(model['sig'])
        self._c = float(model['c'])
        self._std = float(model.get('std', 1))
        desc_siz = model['R_desc'].shape[0]
        n_perms, self._n_atoms = model['perms'].shape
        perm_idxs = torch.tensor(model['tril_perms_lin']).view(-1, n_perms).t()
        self._xs_train, self._Jx_alphas = (
            nn.Parameter(
                xs.repeat(1, n_perms)[:, perm_idxs].reshape(-1, desc_siz),
                requires_grad=False,
            )
            for xs in (
                torch.tensor(model['R_desc']).t(),
                torch.tensor(np.array(model['R_d_desc_alpha'])),
            )
        )

    def forward(self, Rs):
        """
        Predict energy and forces for a batch of geometries.

        Parameters
        ----------
        R : Tensor
            (dims M x N x 3) Cartesian coordinates of M molecules composed of N atoms.

        Returns
        -------
        E : Tensor
            (dims M) Molecular energies.
        F : Tensor
            (dims M x N x 3) Nuclear gradients of the energy.
        """
        assert Rs.dim() == 3
        assert Rs.shape[1:] == (self._n_atoms, 3)
        sig = self._sig
        diffs = Rs[:, :, None, :] - Rs[:, None, :, :]
        dists = diffs.norm(dim=-1)
        i, j = np.diag_indices(self._n_atoms)
        dists[:, i, j] = np.inf
        i, j = np.tril_indices(self._n_atoms, k=-1)
        xs = 1 / dists[:, i, j]
        q = np.sqrt(5) / sig
        x_diffs = (q * xs)[:, None, :] - q * self._xs_train
        x_dists = x_diffs.norm(dim=-1)
        exp_xs = 5 / (3 * sig ** 2) * torch.exp(-x_dists)
        dot_x_diff_Jx_alphas = (x_diffs * self._Jx_alphas).sum(dim=-1)
        exp_xs_1_x_dists = exp_xs * (1 + x_dists)
        F1s_x = ((exp_xs * dot_x_diff_Jx_alphas)[..., None] * x_diffs).sum(dim=1)
        F2s_x = exp_xs_1_x_dists @ self._Jx_alphas
        Fs_x = (F1s_x - F2s_x) * self._std
        Fs = ((expand_tril(Fs_x) / dists ** 3)[..., None] * diffs).sum(dim=1)
        Es = (exp_xs_1_x_dists * dot_x_diff_Jx_alphas).sum(dim=-1) / q
        Es = self._c + Es * self._std
        return Es, Fs

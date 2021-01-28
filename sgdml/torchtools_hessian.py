# coding: utf-8
"""sGDML force field with analytic energies, gradients and Hessians"""

__all__ = ['GDMLPredict']

# MIT License
#
# Copyright (c) 2019-2020 Jan Hermann, Stefan Chmiela
# modified by Alexander Humeniuk
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
import numpy as np
import torch
import torch.nn as nn


class GDMLPredict(nn.Module):
    def __init__(self, model):
        """
        Predict molecular energies, gradients and Hessians from machine-learned GDML model.

        This is a modified version of Stefan Chmiela's GDML potential (adapted from https://github.com/stefanch/sGDML)
        which can also predict second order derivatives (Hessians) of the potential energy.

        Parameters
        ----------
        model : Mapping
            Obtained from :meth:`~train.GDMLTrain.train`. 
            It is assumed that the model uses atomic units (bohr for lengths and Hartree for energies).

        Notes
        -----
        On a CPU, a Hessian calculation is roughly 5 times more expensive than a gradient calculation. On a GPU, it is 10 times
        more expensive than a gradient calculation. Batches of approximately 10000 medium-sized molecules can be computed per second
        on a GPU. 
        """
        super().__init__()

        model = dict(model)

        self._sig = int(model['sig'])
        self._c = float(model['c'])
        self._std = float(model.get('std', 1))

        self.n_atoms = model['z'].shape[0]

        desc_siz = model['R_desc'].shape[0]
        n_perms, self._n_atoms = model['perms'].shape
        perm_idxs = (
            torch.tensor(model['tril_perms_lin'])
            .view(-1, n_perms)
            .t()
        )

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

        self.perm_idxs = perm_idxs
        self.n_perms = n_perms

    def forward(self, r, order=2):
        """
        Predict 
          * molecular energy, 
          * gradient of the energy,
          * matrix of second order derivatives (Hessian) of energy
        for a batch of geometries.

        Parameters
        ----------
        r     :   Tensor
           (dims B x 3N) Cartesian coordinates (in bohr) of M molecules composed of N atoms

        Returns
        -------
        energy  :  Tensor 
           (dims B) Molecular energies (in Hartree)
        grad  :  Tensor 
           (dims B x 3N) Gradients of molecular energies (in Hartree * bohr^{-1})
        hess  :  Tensor 
           (dims B x 3N x 3N) Hessian matrices (in Hartree * bohr^{-2})

        What is returned depends on the option `order`.


        Optional
        --------
        order   :   int
          Order of the highest derivative that will be returned (0 - E, 1 - dE/dx, 2 - d^2E/dxdy).
          Allows to stop the calculation early, if only the energy and/or gradient
          is needed. Depending on the value of `order` 1,2 or 3 Tensors are returned:
            order=0       -     energy
            order=1       -    (energy, grad)
            order=2       -    (energy, grad, hess)
        """
        # dimensions
        #  B: batch size, number of molecular for which predictions should be made
        #  M: number of training samples
        #  N: number of atoms
        #  D: dimension of descriptor, for Coulomb matrix D = N*(N-1)/2
        #  X: spatial dimensions, 3
        #  

        # dimensions
        dimN = self._n_atoms
        dimM, dimD = self._Jx_alphas.size()
        dimB = r.size()[0]
        assert r.size()[1] == 3*dimN

        sig = self._sig
        q = np.sqrt(5) / sig

        r = torch.reshape(r, (-1, dimN, 3))
        # 
        diffs = r[:, :, None, :] - r[:, None, :, :]  # (B, N, N, 3)
        
        dists = diffs.norm(dim=-1)  # (B, N, N)

        # indices of lower tiangular matrix, i > j
        i,j = np.tril_indices(dimN, k=-1)
        xs = 1.0 / dists[:, i, j]   # (B, D)

        del dists

        x_diffs = xs[:, None, :] - self._xs_train  #  (B, M, D)
        x_dists = x_diffs.norm(dim=-1)             #  (B, M)

        A = self._Jx_alphas        
        # XA = sum_n (x_n - x_n') A_n 
        XA = torch.einsum('bmd,md->bm', x_diffs, A)  #  (B, M)

        exp_fac = 1.0/3.0 * q**4 * torch.exp(-q * x_dists)         #  (B, M)
        
        energy = torch.einsum('bm,bm->b', exp_fac * (1.0 + q*x_dists)/q**2, XA)
        energy = (energy * self._std + self._c)

        if order == 0:
            # 0-th order derivative 
            return energy

        #    construct gradient of molecular energy
        
        xs3 = xs**3
        # chain rule: gradient w/r/t descriptor -->  gradient w/r/t cartesian coordinates

        # construct Jacobian of Coulomb matrix D_(ij) = 1/|r(i)-r(j)|
        jacobian = torch.zeros(dimB, dimD, dimN, 3,
                               dtype=diffs.dtype,
                               device=A.device)   #  (B, D, N, 3)
        k,l = torch.tril_indices(dimN, dimN, offset=-1)
        kl = torch.arange(dimD)
        jacobian[:,kl,k,:] = -xs3[:,:,None] * diffs[:,k,l,:]
        jacobian[:,kl,l,:] -= xs3[:,:,None] * diffs[:,l,k,:]
        jacobian = torch.reshape(jacobian, (dimB, dimD, 3*dimN))

        grad_x  = torch.einsum('bm,md->bd', exp_fac * (1.0 + q*x_dists)/q**2, A)  # (B, D)
        grad_x -= torch.einsum('bm,bmd->bd', exp_fac * XA, x_diffs)
        # transform gradient to cartesian coordinates
        grad = torch.einsum('bd,bdx->bx', grad_x, jacobian)
        grad *= self._std

        if order == 1:
            # 0-th and 1st order derivatives
            return energy, grad

        #    construct Hessian of molecular energy
        # XJ = sum_a (x_a - x_a') J_ax
        XJ = torch.einsum('bmd,bdx->bmx', x_diffs, jacobian)      #  (B, M, 3*N)
        AJ = torch.einsum('md,bdx->bmx', A, jacobian)             #  (B, M, 3*N)
        JJ = torch.einsum('bdx,bdy->bxy', jacobian, jacobian)     #  (B, 3*N, 3*N)

        del jacobian

        # sum over training set (M dimension)
        hess  = torch.einsum('bm,bmx,bmy->bxy',
                             exp_fac * XA * q/x_dists, XJ, XJ)     #  (B, 3*N, 3*N)
        hess -= torch.einsum('bm,bxy->bxy', exp_fac * XA, JJ)
        hess -= torch.einsum('bm,bmx,bmy->bxy', exp_fac, AJ, XJ)
        hess -= torch.einsum('bm,bmx,bmy->bxy', exp_fac, XJ, AJ)

        del XA, XJ, AJ, JJ, x_dists, exp_fac

        h1 = ( 3 * grad_x[:,kl,None,None] * xs[:,kl,None,None]**5
                        * diffs[:,k,l,:,None] * diffs[:,k,l,None,:] )
        h2 = -grad_x[:,kl] * xs[:,kl]**3

        idxB = torch.arange(dimB).unsqueeze(1).expand(dimB, dimD)
        # loop over cartesian coordinates u,v = x,y,z
        for u in [0,1,2]:
            for v in [0,1,2]:
                h1_uv = h1[:,:,u,v]
                hess[:,3*k+u,3*l+v] -= h1_uv
                hess[:,3*l+u,3*k+v] -= h1_uv

                # Because the index arrays k and l contain repeated indices
                # we cannot simply use the notation
                #  hess[:,3*k+u,3*k+v] += h1[:,kl,u,v]
                #  hess[:,3*l+u,3*l+v] += h1[:,kl,u,v]
                # since the the contributions from repeated indices are not
                # accumulated. Instead we have to use `index_put_(..., accumulate=True)`.
                hess.index_put_((idxB,3*k+u,3*k+v), h1[:,kl,u,v], accumulate=True)
                hess.index_put_((idxB,3*l+u,3*l+v), h1[:,kl,u,v], accumulate=True)            

            hess[:,3*k+u,3*l+u] -= h2
            hess[:,3*l+u,3*k+u] -= h2

            hess.index_put_((idxB,3*k+u,3*k+u), h2[:,kl], accumulate=True)
            hess.index_put_((idxB,3*l+u,3*l+u), h2[:,kl], accumulate=True)
 
        hess *= self._std

        # 0-th, 1st and 2nd order derivatives
        return energy, grad, hess

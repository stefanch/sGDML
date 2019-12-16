#!/usr/bin/python

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

import multiprocessing as mp
import sys
from functools import partial

import numpy as np
import scipy.optimize
import scipy.spatial.distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from . import ui

glob = {}


def share_array(arr_np, typecode):
    arr = mp.RawArray(typecode, arr_np.ravel())
    return arr, arr_np.shape


def _bipartite_match_wkr(i, n_train, same_z_cost):

    global glob

    adj_set = np.frombuffer(glob['adj_set']).reshape(glob['adj_set_shape'])
    v_set = np.frombuffer(glob['v_set']).reshape(glob['v_set_shape'])
    match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])

    adj_i = scipy.spatial.distance.squareform(adj_set[i, :])
    v_i = v_set[i, :, :]

    match_perms = {}
    for j in range(i + 1, n_train):

        adj_j = scipy.spatial.distance.squareform(adj_set[j, :])
        v_j = v_set[j, :, :]

        cost = -np.fabs(v_i).dot(np.fabs(v_j).T)
        cost += same_z_cost * np.max(np.abs(cost))

        _, perm = scipy.optimize.linear_sum_assignment(cost)

        adj_i_perm = adj_i[:, perm]
        adj_i_perm = adj_i_perm[perm, :]

        score_before = np.linalg.norm(adj_i - adj_j)
        score = np.linalg.norm(adj_i_perm - adj_j)

        match_cost[i, j] = score
        if score >= score_before:
            match_cost[i, j] = score_before
        elif not np.isclose(score_before, score):  # otherwise perm is identity
            match_perms[i, j] = perm

    return match_perms


def bipartite_match(R, z, max_processes=None):

    global glob

    n_train, n_atoms, _ = R.shape

    # penalty matrix for mixing atom species
    same_z_cost = np.repeat(z[:, None], len(z), axis=1) - z
    same_z_cost[same_z_cost != 0] = 1

    match_cost = np.zeros((n_train, n_train))

    adj_set = np.empty((n_train, (n_atoms ** 2 - n_atoms) // 2))
    v_set = np.empty((n_train, n_atoms, n_atoms))
    for i in range(n_train):
        r = np.squeeze(R[i, :, :])

        adj = scipy.spatial.distance.pdist(r, 'euclidean')
        w, v = np.linalg.eig(scipy.spatial.distance.squareform(adj))
        v = v[:, w.argsort()[::-1]]

        adj_set[i, :] = adj
        v_set[i, :, :] = v

    glob['adj_set'], glob['adj_set_shape'] = share_array(adj_set, 'd')
    glob['v_set'], glob['v_set_shape'] = share_array(v_set, 'd')
    glob['match_cost'], glob['match_cost_shape'] = share_array(match_cost, 'd')

    pool = mp.Pool(max_processes)
    match_perms_all = {}
    for i, match_perms in enumerate(
        pool.imap_unordered(
            partial(_bipartite_match_wkr, n_train=n_train, same_z_cost=same_z_cost),
            list(range(n_train)),
        )
    ):
        match_perms_all.update(match_perms)
        ui.progr_bar(i, n_train - 1, disp_str='Bi-partite matching...')
    pool.close()

    match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])
    match_cost = match_cost + match_cost.T
    match_cost[np.diag_indices_from(match_cost)] = np.inf
    match_cost = csr_matrix(match_cost)

    return match_perms_all, match_cost


def sync_perm_mat(match_perms_all, match_cost, n_atoms):

    tree = minimum_spanning_tree(match_cost, overwrite=True)

    perms = np.arange(n_atoms, dtype=int)[None, :]
    rows, cols = tree.nonzero()
    for com in zip(rows, cols):
        perm = match_perms_all.get(com)
        if perm is not None:
            perms = np.vstack((perms, perm))
    perms = np.unique(perms, axis=0)
    ui.progr_toggle(is_done=True, disp_str='Permutation synchronization')

    return perms


def complete_sym_group(perms):

    perm_added = True
    while perm_added:
        perm_added = False
        n_perms = perms.shape[0]
        for i in range(n_perms):
            for j in range(n_perms):

                new_perm = perms[i, perms[j, :]]
                if not (new_perm == perms).all(axis=1).any():
                    perm_added = True
                    perms = np.vstack((perms, new_perm))

    ui.progr_toggle(
        is_done=True,
        disp_str='Symmetry group completion',
        sec_disp_str='{:d} symmetries'.format(perms.shape[0]),
    )

    return perms


def find_perms(R, z, max_processes=None):

    n_atoms = len(z)

    # find matching for all pairs
    match_perms_all, match_cost = bipartite_match(R, z, max_processes)

    # remove inconsistencies
    match_perms = sync_perm_mat(match_perms_all, match_cost, n_atoms)

    # commplete symmetric group
    sym_group_perms = complete_sym_group(match_perms)

    return sym_group_perms


def inv_perm(perm):

    inv_perm = np.empty(perm.size, perm.dtype)
    inv_perm[perm] = np.arange(perm.size)

    return inv_perm

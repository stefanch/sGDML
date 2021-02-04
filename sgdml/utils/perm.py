#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2021 Stefan Chmiela
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

Pool = mp.get_context('fork').Pool

import sys
import timeit
from functools import partial

import numpy as np
import scipy.optimize
import scipy.spatial.distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .. import DONE, NOT_DONE
from .desc import Desc
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


def bipartite_match(R, z, lat_and_inv=None, max_processes=None, callback=None):

    global glob

    n_train, n_atoms, _ = R.shape

    # penalty matrix for mixing atom species
    same_z_cost = np.repeat(z[:, None], len(z), axis=1) - z
    same_z_cost[same_z_cost != 0] = 1




    # NEW

    # penalty matrix for mixing differently bonded atoms
    # NOTE: needs ASE, expects R to be in angstrom, does not support bond breaking

    # from ase import Atoms
    # from ase.geometry.analysis import Analysis

    # atoms = Atoms(
    #     z, positions=R[0]
    # )  # only use first molecule in dataset to find connected components (fix me later, maybe) # *0.529177249

    # bonds = Analysis(atoms).all_bonds[0]
    # #n_bonds = np.array([len(bonds_i) for bonds_i in bonds])

    # same_bonding_cost = np.zeros((n_atoms, n_atoms))
    # for i in range(n_atoms):
    #     bi = bonds[i]
    #     z_bi = z[bi]
    #     for j in range(i+1,n_atoms):
    #         bj = bonds[j]
    #         z_bj = z[bj]

    #         if set(z_bi) == set(z_bj):
    #             same_bonding_cost[i,j] = 1


    # same_bonding_cost += same_bonding_cost.T
    
    # same_bonding_cost[np.diag_indices(n_atoms)] = 1
    # same_bonding_cost = 1-same_bonding_cost


    #set(a) & set(b)

    #same_bonding_cost = np.repeat(n_bonds[:, None], len(n_bonds), axis=1) - n_bonds
    #same_bonding_cost[same_bonding_cost != 0] = 1



    # NEW




    match_cost = np.zeros((n_train, n_train))

    desc = Desc(n_atoms, max_processes=max_processes)

    adj_set = np.empty((n_train, desc.dim))
    v_set = np.empty((n_train, n_atoms, n_atoms))
    for i in range(n_train):
        r = np.squeeze(R[i, :, :])

        if lat_and_inv is None:
            adj = scipy.spatial.distance.pdist(r, 'euclidean')


            # from ase import Atoms
            # from ase.geometry.analysis import Analysis

            # atoms = Atoms(
            #     z, positions=r
            # )  # only use first molecule in dataset to find connected components (fix me later, maybe) # *0.529177249

            # bonds = Analysis(atoms).all_bonds[0]
            
            #adj = scipy.spatial.distance.squareform(adj)

            #bonded = np.zeros((z.size, z.size))

            #for j, bonded_to in enumerate(bonds):
                #inv_bonded_to = np.arange(n_atoms)
                #inv_bonded_to[bonded_to] = 0

                #adj[j, inv_bonded_to] = 0

            #    bonded[j, bonded_to] = 1

            # bonded = bonded + bonded.T

            # print(bonded)

        else:
            adj = scipy.spatial.distance.pdist(
                r, lambda u, v: np.linalg.norm(desc.pbc_diff(u - v, lat_and_inv))
            )

        w, v = np.linalg.eig(scipy.spatial.distance.squareform(adj))
        v = v[:, w.argsort()[::-1]]

        adj_set[i, :] = adj
        v_set[i, :, :] = v

    glob['adj_set'], glob['adj_set_shape'] = share_array(adj_set, 'd')
    glob['v_set'], glob['v_set_shape'] = share_array(v_set, 'd')
    glob['match_cost'], glob['match_cost_shape'] = share_array(match_cost, 'd')

    if callback is not None:
        callback = partial(callback, disp_str='Bi-partite matching')

    start = timeit.default_timer()
    pool = Pool(max_processes)

    match_perms_all = {}
    for i, match_perms in enumerate(
        pool.imap_unordered(
            partial(_bipartite_match_wkr, n_train=n_train, same_z_cost=same_z_cost),
            list(range(n_train)),
        )
    ):
        match_perms_all.update(match_perms)

        if callback is not None:
            callback(i, n_train)

    pool.close()
    pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
    stop = timeit.default_timer()

    dur_s = (stop - start) / 2
    sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
    if callback is not None:
        callback(n_train, n_train, sec_disp_str=sec_disp_str)

    match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])
    match_cost = match_cost + match_cost.T
    match_cost[np.diag_indices_from(match_cost)] = np.inf
    match_cost = csr_matrix(match_cost)

    return match_perms_all, match_cost


def sync_perm_mat(match_perms_all, match_cost, n_atoms, callback=None):

    if callback is not None:
        callback = partial(
            callback, disp_str='Multi-partite matching (permutation synchronization)'
        )
        callback(NOT_DONE)

    tree = minimum_spanning_tree(match_cost, overwrite=True)

    perms = np.arange(n_atoms, dtype=int)[None, :]
    rows, cols = tree.nonzero()
    for com in zip(rows, cols):
        perm = match_perms_all.get(com)
        if perm is not None:
            perms = np.vstack((perms, perm))
    perms = np.unique(perms, axis=0)

    if callback is not None:
        callback(DONE)

    return perms

# convert permutation to dijoined cycles
def to_cycles(perm):
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi)) # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break

        cycles.append(cycle)

    return cycles

# find permutation group with larges cardinality
# note: this is used if transitive closure fails (to salvage at least some permutations)
def salvage_subgroup(perms):

    n_perms, n_atoms = perms.shape
    lcms = []
    for i in range(n_perms):
        cy_lens = [len(cy) for cy in to_cycles(list(perms[i, :]))]
        lcm = np.lcm.reduce(cy_lens)
        lcms.append(lcm)
    keep_idx = np.argmax(lcms)
    perms = np.vstack((np.arange(n_atoms), perms[keep_idx,:]))

    return perms


def complete_sym_group(perms, n_perms_max=None, disp_str='Permutation group completion', callback=None):

    if callback is not None:
        callback = partial(callback, disp_str=disp_str)
        callback(NOT_DONE)

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

                    # Transitive closure is not converging! Give up and return identity permutation.
                    if n_perms_max is not None and perms.shape[0] == n_perms_max:

                        if callback is not None:
                            callback(
                                DONE,
                                sec_disp_str='transitive closure has failed',
                                done_with_warning=True,
                            )
                        return None

    if callback is not None:
        callback(
            DONE,
            sec_disp_str='found {:d} symmetries'.format(perms.shape[0]),
        )

    return perms


def find_perms(R, z, lat_and_inv=None, callback=None, max_processes=None):

    m, n_atoms = R.shape[:2]

    # Find matching for all pairs.
    match_perms_all, match_cost = bipartite_match(
        R, z, lat_and_inv, max_processes, callback=callback
    )

    # Remove inconsistencies.
    match_perms = sync_perm_mat(match_perms_all, match_cost, n_atoms, callback=callback)

    # Commplete symmetric group.
    # Give up, if transitive closure yields more than 100 unique permutations.
    sym_group_perms = complete_sym_group(match_perms, n_perms_max=100, callback=callback)

    # Limit closure to largest cardinality permutation in the set to get at least some symmetries.
    if sym_group_perms is None:
        match_perms_subset = salvage_subgroup(match_perms)
        sym_group_perms = complete_sym_group(match_perms_subset, n_perms_max=100, disp_str='Closure disaster recovery', callback=callback)

    return sym_group_perms


def find_frag_perms(R, z, lat_and_inv=None, callback=None, max_processes=None):

    from ase import Atoms
    from ase.geometry.analysis import Analysis
    from scipy.sparse.csgraph import connected_components

    print('Finding permutable non-bonded fragments... (assumes Ang!)')

    # TODO: positions must be in Angstrom for this to work!!

    n_train, n_atoms = R.shape[:2]
    atoms = Atoms(
        z, positions=R[0]
    )  # only use first molecule in dataset to find connected components (fix me later, maybe) # *0.529177249

    adj = Analysis(atoms).adjacency_matrix[0]
    _, labels = connected_components(csgraph=adj, directed=False, return_labels=True)

    frags = []
    for label in np.unique(labels):
        frags.append(np.where(labels == label)[0])
    n_frags = len(frags)

    if n_frags == n_atoms:
        print(
            'Skipping fragment symmetry search (something went wrong, e.g. length unit not in Angstroms, etc.)'
        )
        return [range(n_atoms)]

    # print(labels)

    # from . import ui, io
    # xyz_str = io.generate_xyz_str(R[0][np.where(labels == 0)[0], :]*0.529177249, z[np.where(labels == 0)[0]])
    # xyz_str = ui.indent_str(xyz_str, 2)
    # sprint(xyz_str)

    # NEW

    # uniq_labels = np.unique(labels)
    # R_cg = np.empty((R.shape[0], len(uniq_labels), R.shape[2]))
    # z_frags = []
    # z_cg = []
    # for label in uniq_labels:
    #     frag_idxs = np.where(labels == label)[0]

    #     R_cg[:,label,:] = np.mean(R[:,frag_idxs,:], axis=1)
    #     z_frag = np.sort(z[frag_idxs])

    #     z_frag_label = 0
    #     if len(z_frags) == 0:
    #         z_frags.append(z_frag)
    #     else:
    #         z_frag_label = np.where(np.all(z_frags == z_frag, axis=1))[0]

    #         if len(z_frag_label) == 0: # not found
    #             z_frag_label = len(z_frags)
    #             z_frags.append(z_frag)
    #         else:
    #             z_frag_label = z_frag_label[0]

    #     z_cg.append(z_frag_label)

    # print(z_cg)
    # print(R_cg.shape)

    # perms = find_perms(R_cg, np.array(z_cg), lat_and_inv=lat_and_inv, max_processes=max_processes)

    # print('cg perms')
    # print(perms)

    # NEW

    # print(n_frags)

    print('| Found ' + str(n_frags) + ' disconnected fragments.')

    n_frags_unique = 0 # number of unique fragments

    # match fragments to find identical ones (allows permutations of fragments)
    swap_perms = [np.arange(n_atoms)]
    for f1 in range(n_frags):
        for f2 in range(f1 + 1, n_frags):

            sort_idx_f1 = np.argsort(z[frags[f1]])
            sort_idx_f2 = np.argsort(z[frags[f2]])
            inv_sort_idx_f2 = inv_perm(sort_idx_f2)

            z1 = z[frags[f1]][sort_idx_f1]
            z2 = z[frags[f2]][sort_idx_f2]

            if np.array_equal(z1, z2): # fragment have the same composition
                n_frags_unique += 1

                for ri in range(
                    min(10, R.shape[0])
                ):  # only use first molecule in dataset for matching (fix me later)

                    R_match1 = R[ri, frags[f1], :]
                    R_match2 = R[ri, frags[f2], :]
                    
                    #if np.array_equal(z1, z2):

                    R_pair = np.concatenate(
                        (R_match1[None, sort_idx_f1, :], R_match2[None, sort_idx_f2, :])
                    )

                    perms = find_perms(
                        R_pair, z1, lat_and_inv=lat_and_inv, max_processes=max_processes
                    )

                    # embed local permutation into global context
                    for p in perms:

                        match_perm = sort_idx_f1[p][inv_sort_idx_f2]

                        swap_perm = np.arange(n_atoms)
                        swap_perm[frags[f1]] = frags[f2][match_perm]
                        swap_perm[frags[f2][match_perm]] = frags[f1]
                        swap_perms.append(swap_perm)

    swap_perms = np.unique(np.array(swap_perms), axis=0)


    print('| Found ' + str(n_frags_unique) + ' (likely to be) *unique* disconnected fragments.')

    # commplete symmetric group
    sym_group_perms = complete_sym_group(swap_perms)
    print(
        '| Found '
        + str(sym_group_perms.shape[0])
        + ' fragment permutations after closure.'
    )

    # match fragments with themselves (to find symmetries in each fragment)
    if n_frags > 1:
        print('| Matching individual fragments.')
        for f in range(n_frags):

            R_frag = R[:, frags[f], :]
            z_frag = z[frags[f]]

            # print(R_frag.shape)
            # print(z_frag)
            print(f)

            perms = find_perms(
                R_frag, z_frag, lat_and_inv=lat_and_inv, max_processes=max_processes
            )

            # print(f)
            print(perms)



    f = 0
    perms = find_perms_via_alignment(R[0, :, :], frags[f], [215, 214, 210, 211], [209, 208, 212, 213], z, lat_and_inv=lat_and_inv, max_processes=max_processes)
    #perms = find_perms_via_alignment(R[0, :, :], frags[f], [214, 215, 210, 211], [209, 208, 212, 213], z, lat_and_inv=lat_and_inv, max_processes=max_processes)
    sym_group_perms = np.vstack((perms[None,:], sym_group_perms))
    sym_group_perms = complete_sym_group(sym_group_perms, callback=callback)

    #print(sym_group_perms.shape)

    #import sys
    #sys.exit()

    return sym_group_perms


def find_perms_via_alignment(pts_full, frag_idxs, align_a_idxs, align_b_idxs, z, lat_and_inv=None, max_processes=None):

    # 1. find rotatino that aligns points (Nx3 matrix) in 'align_a_idxs' with points in 'align_b_idxs' 
    # 2. rotate the whole thing
    # find perms by matching those two structures

    #align_a_ctr = np.mean(align_a_pts, axis=0)
    #align_b_ctr = np.mean(align_b_pts, axis=0)

    pts = pts_full[frag_idxs, :]

    align_a_pts = pts[align_a_idxs,:]
    align_b_pts = pts[align_b_idxs,:]

    ctr = np.mean(pts, axis=0)
    align_a_pts -= ctr
    align_b_pts -= ctr

    ab_cov = align_a_pts.T.dot(align_b_pts)
    u, s, vh = np.linalg.svd(ab_cov)
    R = u.dot(vh)

    if np.linalg.det(R) < 0:
        vh[2,:] *= -1 #multiply 3rd column of V by -1
        R = u.dot(vh)

    pts -= ctr
    pts_R = pts.copy()
    pts_R = R.dot(pts_R.T).T

    pts += ctr
    pts_R += ctr


    pts_full_R = pts_full.copy()
    pts_full_R[frag_idxs, :] = pts_R


    R_pair = np.vstack((pts_full[None,:,:], pts_full_R[None,:,:]))


    #from . import io

    #xyz_str = io.generate_xyz_str(pts_full, z)
    #print(xyz_str)

    #xyz_str = io.generate_xyz_str(pts_full_R, z)
    #print(xyz_str)


    z_frag = z[frag_idxs]



    adj = scipy.spatial.distance.cdist(R_pair[0], R_pair[1], 'euclidean')
    _, perm = scipy.optimize.linear_sum_assignment(adj)

    score_before = np.linalg.norm(adj)

    adj_perm = scipy.spatial.distance.cdist(R_pair[0,:], R_pair[0, perm], 'euclidean')
    score = np.linalg.norm(adj_perm)



    #perms = find_perms(
    #    R_pair, z, lat_and_inv=lat_and_inv, max_processes=max_processes
    #)

    return perm


def inv_perm(perm):

    inv_perm = np.empty(perm.size, perm.dtype)
    inv_perm[perm] = np.arange(perm.size)

    return inv_perm

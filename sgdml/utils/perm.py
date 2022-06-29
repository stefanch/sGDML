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

    # set(a) & set(b)

    # same_bonding_cost = np.repeat(n_bonds[:, None], len(n_bonds), axis=1) - n_bonds
    # same_bonding_cost[same_bonding_cost != 0] = 1

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

            # adj = scipy.spatial.distance.squareform(adj)

            # bonded = np.zeros((z.size, z.size))

            # for j, bonded_to in enumerate(bonds):
            # inv_bonded_to = np.arange(n_atoms)
            # inv_bonded_to[bonded_to] = 0

            # adj[j, inv_bonded_to] = 0

            #    bonded[j, bonded_to] = 1

            # bonded = bonded + bonded.T

            # print(bonded)

        else:

            from .desc import _pdist, _squareform

            adj_tri = _pdist(r, lat_and_inv)
            adj = _squareform(adj_tri)  # our vectorized format to full matrix
            adj = scipy.spatial.distance.squareform(
                adj
            )  # full matrix to numpy vectorized format

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

    pool = None
    map_func = map
    if max_processes != 1 and mp.cpu_count() > 1:
        pool = Pool((max_processes or mp.cpu_count()) - 1)  # exclude main process
        map_func = pool.imap_unordered

    match_perms_all = {}
    for i, match_perms in enumerate(
        map_func(
            partial(_bipartite_match_wkr, n_train=n_train, same_z_cost=same_z_cost),
            list(range(n_train)),
        )
    ):
        match_perms_all.update(match_perms)

        if callback is not None:
            callback(i, n_train)

    if pool is not None:
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        pool = None

    stop = timeit.default_timer()

    dur_s = stop - start
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
        elem0 = next(iter(pi))  # arbitrary starting element
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

    all_long_cycles = []
    for i in range(n_perms):
        long_cycles = [cy for cy in to_cycles(list(perms[i, :])) if len(cy) > 1]
        all_long_cycles += long_cycles

    # print(all_long_cycles)
    # print('--------------')

    def _cycle_intersects_with_larger_one(cy):

        for ac in all_long_cycles:
            if len(cy) < len(ac):
                if not set(cy).isdisjoint(ac):
                    return True

        return False

    lcms = []
    keep_idx_many = []
    for i in range(n_perms):

        # print(to_cycles(list(perms[i, :])))

        # is this permutation valid?
        # remove permutations that contain cycles that share elements with larger cycles in other perms
        long_cycles = [cy for cy in to_cycles(list(perms[i, :])) if len(cy) > 1]

        # print('long cycles:')
        # print(long_cycles)

        ignore_perm = any(list(map(_cycle_intersects_with_larger_one, long_cycles)))

        if not ignore_perm:
            keep_idx_many.append(i)

        # print(ignore_perm)

        # print()

        # cy_lens = [len(cy) for cy in to_cycles(list(perms[i, :]))]
        # lcm = np.lcm.reduce(cy_lens)
        # lcms.append(lcm)
    # keep_idx = np.argmax(lcms)
    # perms = np.vstack((np.arange(n_atoms), perms[keep_idx,:]))
    perms = perms[keep_idx_many, :]

    # print(perms)

    return perms


def complete_sym_group(
    perms, n_perms_max=None, disp_str='Permutation group completion', callback=None
):

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
    sym_group_perms = complete_sym_group(
        match_perms, n_perms_max=100, callback=callback
    )

    # Limit closure to largest cardinality permutation in the set to get at least some symmetries.
    if sym_group_perms is None:
        match_perms_subset = salvage_subgroup(match_perms)
        sym_group_perms = complete_sym_group(
            match_perms_subset,
            n_perms_max=100,
            disp_str='Closure disaster recovery',
            callback=callback,
        )

    return sym_group_perms


def find_extra_perms(R, z, lat_and_inv=None, callback=None, max_processes=None):

    m, n_atoms = R.shape[:2]

    # NEW

    # catcher
    # p = np.arange(n_atoms)
    # plane_3idxs = [19,17,47] # left to right
    # perm = find_perms_via_reflection(R[0], z, np.arange(n_atoms), plane_3idxs, lat_and_inv=None, max_processes=None)
    # perms = np.vstack((p[None,:], perm))
    # plane_3idxs = [(4,5),(2,1),(34,33)]  # top to bottom
    # perm = find_perms_via_reflection(R[0], z, np.arange(n_atoms), plane_3idxs, lat_and_inv=None, max_processes=None)
    # perms = np.vstack((perm[None,:], perms))
    # sym_group_perms = complete_sym_group(perms, n_perms_max=100, callback=callback)

    # nanotube
    R = R.copy()
    frags = find_frags(R[0], z, lat_and_inv=lat_and_inv)
    print(frags)

    perms = np.arange(n_atoms)[None, :]

    plane_3idxs = [280, 281, 273]  # half outer
    add_perms = find_perms_via_reflection(
        R[0], z, frags[1], plane_3idxs, lat_and_inv=None, max_processes=None
    )
    perms = np.vstack((perms, add_perms))

    # rotate inner
    # add_perms = find_perms_via_alignment(R[0], frags[0], [214, 215, 210, 211], [209, 208, 212, 213], z, lat_and_inv=lat_and_inv, max_processes=max_processes)
    # perms = np.vstack((perms, add_perms))
    # sym_group_perms = complete_sym_group(perms, callback=callback)

    # rotate outer
    # add_perms = find_perms_via_alignment(R[0], frags[1], [361, 360, 368, 369], [363, 362, 356, 357], z, lat_and_inv=lat_and_inv, max_processes=max_processes)
    # perms = np.vstack((perms, add_perms))
    # sym_group_perms = complete_sym_group(perms, callback=callback)

    perms = np.unique(perms, axis=0)
    sym_group_perms = complete_sym_group(perms, callback=callback)
    print(sym_group_perms.shape)

    return sym_group_perms

    # buckycatcher
    R = R.copy()  # *0.529177
    frags = find_frags(R[0], z, lat_and_inv=lat_and_inv)

    perms = np.arange(n_atoms)[None, :]

    # syms of catcher
    plane_3idxs = [54, 47, 17]  # left to right
    add_perms = find_perms_via_reflection(
        R[0], z, frags[0], plane_3idxs, lat_and_inv=None, max_processes=None
    )
    perms = np.vstack((perms, add_perms))

    plane_3idxs = [(33, 34), (31, 30), (5, 4)]  # top to bottom
    add_perms = find_perms_via_reflection(
        R[0], z, frags[0], plane_3idxs, lat_and_inv=None, max_processes=None
    )
    perms = np.vstack((perms, add_perms))

    # move cells
    # add_perms = find_perms_via_alignment(R[0], frags[1], [128, 129, 127], [133, 132, 134], z, lat_and_inv=lat_and_inv, max_processes=max_processes)
    # perms = np.vstack((perms, add_perms))
    # sym_group_perms = complete_sym_group(perms, callback=callback)

    # print(sym_group_perms.shape)

    # rotate cells
    add_perms = find_perms_via_alignment(
        R[0],
        frags[1],
        [129, 128, 127],
        [128, 127, 135],
        z,
        lat_and_inv=lat_and_inv,
        max_processes=max_processes,
    )
    perms = np.vstack((perms, add_perms))
    # print(add_perms.shape)
    # sym_group_perms = complete_sym_group(perms, callback=callback)

    # rotate cells (triangle)
    # add_perms = find_perms_via_alignment(R[0], frags[1], [132, 129, 134], [129, 134, 132], z, lat_and_inv=lat_and_inv, max_processes=max_processes)
    # perms = np.vstack((perms, add_perms))
    sym_group_perms = complete_sym_group(perms, callback=callback)

    # print(perms.shape)
    print(sym_group_perms.shape)

    # frag 1: bucky ball
    # perms = find_perms_in_frag(R, z, frags[1], lat_and_inv=lat_and_inv, max_processes=max_processes)
    # perms = np.vstack((p[None,:], perms))

    # print('perms')
    # print(perms.shape)

    # perms = np.unique(perms, axis=0)
    # perms = complete_sym_group(perms, callback=callback)

    # print('perms')
    # print(perms.shape)
    # print(sym_group_perms.shape)

    return sym_group_perms

    # NEW


def find_frags(r, z, lat_and_inv=None):

    from ase import Atoms
    from ase.geometry.analysis import Analysis
    from scipy.sparse.csgraph import connected_components

    print('Finding permutable non-bonded fragments... (assumes Ang!)')

    lat = None
    if lat_and_inv:
        lat = lat_and_inv[0]

    n_atoms = r.shape[0]
    atoms = Atoms(
        z, positions=r, cell=lat, pbc=lat is not None
    )  # only use first molecule in dataset to find connected components (fix me later, maybe) # *0.529177249

    adj = Analysis(atoms).adjacency_matrix[0]
    _, labels = connected_components(csgraph=adj, directed=False, return_labels=True)

    # frags = []
    # for label in np.unique(labels):
    #    frags.append(np.where(labels == label)[0])
    frags = [np.where(labels == label)[0] for label in np.unique(labels)]
    n_frags = len(frags)

    if n_frags == n_atoms:
        print(
            'Skipping fragment symmetry search (something went wrong, e.g. length unit not in Angstroms, etc.)'
        )
        return None

    print('| Found ' + str(n_frags) + ' disconnected fragments.')

    return frags


def find_frag_perms(R, z, lat_and_inv=None, callback=None, max_processes=None):

    from ase import Atoms
    from ase.geometry.analysis import Analysis
    from scipy.sparse.csgraph import connected_components

    # TODO: positions must be in Angstrom for this to work!!

    n_train, n_atoms = R.shape[:2]
    lat, lat_inv = lat_and_inv

    atoms = Atoms(
        z, positions=R[0], cell=lat, pbc=lat is not None
    )  # only use first molecule in dataset to find connected components (fix me later, maybe) # *0.529177249

    adj = Analysis(atoms).adjacency_matrix[0]
    _, labels = connected_components(csgraph=adj, directed=False, return_labels=True)

    # frags = []
    # for label in np.unique(labels):
    #    frags.append(np.where(labels == label)[0])
    frags = [np.where(labels == label)[0] for label in np.unique(labels)]
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

    # ufrags = np.unique([np.sort(z[frag]) for frag in frags])
    # print(ufrags)

    # sys.exit()

    # n_frags_unique = 0 # number of unique fragments

    # match fragments to find identical ones (allows permutations of fragments)
    swap_perms = [np.arange(n_atoms)]
    for f1 in range(n_frags):
        for f2 in range(f1 + 1, n_frags):

            sort_idx_f1 = np.argsort(z[frags[f1]])
            sort_idx_f2 = np.argsort(z[frags[f2]])
            inv_sort_idx_f2 = inv_perm(sort_idx_f2)

            z1 = z[frags[f1]][sort_idx_f1]
            z2 = z[frags[f2]][sort_idx_f2]

            if np.array_equal(z1, z2):  # fragment have the same composition

                for ri in range(
                    min(10, R.shape[0])
                ):  # only use first molecule in dataset for matching (fix me later)

                    R_match1 = R[ri, frags[f1], :]
                    R_match2 = R[ri, frags[f2], :]

                    # if np.array_equal(z1, z2):

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

            # else:
            #    n_frags_unique += 1

    swap_perms = np.unique(np.array(swap_perms), axis=0)

    # print(swap_perms)

    # print('| Found ' + str(n_frags_unique) + ' (likely to be) *unique* disconnected fragments.')

    # commplete symmetric group
    sym_group_perms = complete_sym_group(swap_perms)
    print(
        '| Found '
        + str(sym_group_perms.shape[0])
        + ' fragment permutations after closure.'
    )

    # return sym_group_perms

    # match fragments with themselves (to find symmetries in each fragment)

    def _frag_perm_to_perm(n_atoms, frag_idxs, frag_perms):

        # frag_idxs - indices of the fragment (one fragment!)
        # frag_perms - N fragment permutations (Nxn_atoms)

        perms = np.arange(n_atoms)[None, :]
        for fp in frag_perms:

            p = np.arange(n_atoms)
            p[frag_idxs] = frag_idxs[fp]
            perms = np.vstack((p[None, :], perms))

        return perms

    if n_frags > 1:
        print('| Finding symmetries in individual fragments.')
        for f in range(n_frags):

            R_frag = R[:, frags[f], :]
            z_frag = z[frags[f]]

            frag_perms = find_perms(
                R_frag, z_frag, lat_and_inv=lat_and_inv, max_processes=max_processes
            )

            perms = _frag_perm_to_perm(n_atoms, frags[f], frag_perms)
            sym_group_perms = np.vstack((perms, sym_group_perms))

            print('{:d} perms'.format(perms.shape[0]))

        sym_group_perms = np.unique(sym_group_perms, axis=0)
    sym_group_perms = complete_sym_group(sym_group_perms, callback=callback)

    return sym_group_perms

    # f = 0
    # perms = find_perms_via_alignment(R[0, :, :], frags[f], [215, 214, 210, 211], [209, 208, 212, 213], z, lat_and_inv=lat_and_inv, max_processes=max_processes)
    # #perms = find_perms_via_alignment(R[0, :, :], frags[f], [214, 215, 210, 211], [209, 208, 212, 213], z, lat_and_inv=lat_and_inv, max_processes=max_processes)
    # sym_group_perms = np.vstack((perms[None,:], sym_group_perms))
    # sym_group_perms = complete_sym_group(sym_group_perms, callback=callback)

    # #print(sym_group_perms.shape)

    # #import sys
    # #sys.exit()

    # return sym_group_perms


def _frag_perm_to_perm(n_atoms, frag_idxs, frag_perms):

    # frag_idxs - indices of the fragment (one fragment!)
    # frag_perms - N fragment permutations (Nxn_atoms)

    perms = np.arange(n_atoms)[None, :]
    for fp in frag_perms:

        p = np.arange(n_atoms)
        p[frag_idxs] = frag_idxs[fp]
        perms = np.vstack((p[None, :], perms))

    return perms


def find_perms_in_frag(R, z, frag_idxs, lat_and_inv=None, max_processes=None):

    n_atoms = R.shape[1]

    R_frag = R[:, frag_idxs, :]
    z_frag = z[frag_idxs]

    frag_perms = find_perms(
        R_frag, z_frag, lat_and_inv=lat_and_inv, max_processes=max_processes
    )

    perms = _frag_perm_to_perm(n_atoms, frag_idxs, frag_perms)

    return perms


def find_perms_via_alignment(
    pts_full,
    frag_idxs,
    align_a_idxs,
    align_b_idxs,
    z,
    lat_and_inv=None,
    max_processes=None,
):

    # 1. find rotatino that aligns points (Nx3 matrix) in 'align_a_idxs' with points in 'align_b_idxs'
    # 2. rotate the whole thing
    # find perms by matching those two structures (match atoms that are closest after transformation)

    # align_a_ctr = np.mean(align_a_pts, axis=0)
    # align_b_ctr = np.mean(align_b_pts, axis=0)

    # alignment indices are included in fragment
    assert np.isin(align_a_idxs, frag_idxs).all()
    assert np.isin(align_b_idxs, frag_idxs).all()

    assert len(align_a_idxs) == len(align_b_idxs)

    # align_a_frag_idxs = np.where(np.in1d(frag_idxs, align_a_idxs))[0]
    # align_b_frag_idxs = np.where(np.in1d(frag_idxs, align_b_idxs))[0]

    pts = pts_full[frag_idxs, :]

    align_a_pts = pts_full[align_a_idxs, :]
    align_b_pts = pts_full[align_b_idxs, :]

    ctr = np.mean(pts, axis=0)
    align_a_pts -= ctr
    align_b_pts -= ctr

    ab_cov = align_a_pts.T.dot(align_b_pts)
    u, s, vh = np.linalg.svd(ab_cov)
    R = u.dot(vh)

    if np.linalg.det(R) < 0:
        vh[2, :] *= -1  # multiply 3rd column of V by -1
        R = u.dot(vh)

    pts -= ctr
    pts_R = pts.copy()

    pts_R = R.dot(pts_R.T).T

    pts += ctr
    pts_R += ctr

    pts_full_R = pts_full.copy()
    pts_full_R[frag_idxs, :] = pts_R

    R_pair = np.vstack((pts_full[None, :, :], pts_full_R[None, :, :]))

    # from . import io

    # xyz_str = io.generate_xyz_str(pts_full, z)
    # print(xyz_str)

    # xyz_str = io.generate_xyz_str(pts_full_R, z)
    # print(xyz_str)

    # z_frag = z[frag_idxs]

    adj = scipy.spatial.distance.cdist(R_pair[0], R_pair[1], 'euclidean')
    _, perm = scipy.optimize.linear_sum_assignment(adj)

    # score_before = np.linalg.norm(adj)

    # adj_perm = scipy.spatial.distance.cdist(R_pair[0,:], R_pair[0, perm], 'euclidean')
    # score = np.linalg.norm(adj_perm)

    # print(score_before)
    # print(score)

    # print('---')

    # print('data \'model example\'', '|', end='')
    # rint('testing', '|', end='')
    # n_atoms = pts_full.shape[1]
    # print(n_atoms)

    # for p in pts_full[:,:]:
    #    print('H {:.5f} {:.5f} {:.5f}'.format(*p), '|', end='')

    # print('end \'model example\';show data')

    # draw selection
    if False:

        print('---')

        from matplotlib import cm

        viridis = cm.get_cmap('prism')
        colors = viridis(np.linspace(0, 1, len(align_a_idxs)))

        for i, idx in enumerate(align_a_idxs):
            color_str = (
                '['
                + str(int(colors[i, 0] * 255))
                + ','
                + str(int(colors[i, 1] * 255))
                + ','
                + str(int(colors[i, 2] * 255))
                + ']'
            )
            print('select atomno=' + str(idx + 1) + '; color ' + color_str)

        for i, idx in enumerate(align_b_idxs):
            color_str = (
                '['
                + str(int(colors[i, 0] * 255))
                + ','
                + str(int(colors[i, 1] * 255))
                + ','
                + str(int(colors[i, 2] * 255))
                + ']'
            )
            print('select atomno=' + str(idx + 1) + '; color ' + color_str)
        print('---')

    return perm


def find_perms_via_reflection(
    r, z, frag_idxs, plane_3idxs, lat_and_inv=None, max_processes=None
):

    # plane_3idxs can be tuples of atoms (to take their center) or atom indices

    # pts = pts_full[frag_idxs, :]
    # pts = r.copy()

    # compute normal of plane defined by atoms in 'plane_idxs'

    is_plane_defined_by_bond_centers = type(plane_3idxs[0]) is tuple
    if is_plane_defined_by_bond_centers:
        a = (r[plane_3idxs[0][0], :] + r[plane_3idxs[0][1], :]) / 2
        b = (r[plane_3idxs[1][0], :] + r[plane_3idxs[1][1], :]) / 2
        c = (r[plane_3idxs[2][0], :] + r[plane_3idxs[2][1], :]) / 2
    else:
        a = r[plane_3idxs[0], :]
        b = r[plane_3idxs[1], :]
        c = r[plane_3idxs[2], :]

    ab = b - a
    ab /= np.linalg.norm(ab)

    ac = c - a
    ac /= np.linalg.norm(ac)

    normal = np.cross(ab, ac)[:, None]

    # compute reflection matrix
    reflection = np.eye(3) - 2 * normal.dot(normal.T)

    r_R = r.copy()
    r_R[frag_idxs, :] = reflection.dot(r[frag_idxs, :].T).T

    # R_pair = np.vstack((r[None,:,:], r_R[None,:,:]))

    adj = scipy.spatial.distance.cdist(r, r_R, 'euclidean')
    _, perm = scipy.optimize.linear_sum_assignment(adj)

    print_perm_colors(perm, r, plane_3idxs)

    # score_before = np.linalg.norm(adj)

    # adj_perm = scipy.spatial.distance.cdist(R_pair[0,:], R_pair[0, perm], 'euclidean')
    # score = np.linalg.norm(adj_perm)

    return perm


def print_perm_colors(perm, pts, plane_3idxs=None):

    idx_done = []
    c = -1
    for i in range(perm.shape[0]):
        if i not in idx_done and perm[i] not in idx_done:
            c += 1
            idx_done += [i]
            idx_done += [perm[i]]

    from matplotlib import cm

    viridis = cm.get_cmap('prism')
    colors = viridis(np.linspace(0, 1, c + 1))

    print('---')
    print('select all; color [255,255,255]')

    if plane_3idxs is not None:

        def pts_str(x):
            return '{' + str(x[0]) + ', ' + str(x[1]) + ', ' + str(x[2]) + '}'

        is_plane_defined_by_bond_centers = type(plane_3idxs[0]) is tuple
        if is_plane_defined_by_bond_centers:
            a = (pts[plane_3idxs[0][0], :] + pts[plane_3idxs[0][1], :]) / 2
            b = (pts[plane_3idxs[1][0], :] + pts[plane_3idxs[1][1], :]) / 2
            c = (pts[plane_3idxs[2][0], :] + pts[plane_3idxs[2][1], :]) / 2
        else:
            a = pts[plane_3idxs[0], :]
            b = pts[plane_3idxs[1], :]
            c = pts[plane_3idxs[2], :]

        print(
            'draw plane1 300 PLANE '
            + pts_str(a)
            + ' '
            + pts_str(b)
            + ' '
            + pts_str(c)
            + ';color $plane1 green'
        )

    idx_done = []
    c = -1
    for i in range(perm.shape[0]):
        if i not in idx_done and perm[i] not in idx_done:

            c += 1
            color_str = (
                '['
                + str(int(colors[c, 0] * 255))
                + ','
                + str(int(colors[c, 1] * 255))
                + ','
                + str(int(colors[c, 2] * 255))
                + ']'
            )

            if i != perm[i]:
                print('select atomno=' + str(i + 1) + '; color ' + color_str)
                print('select atomno=' + str(perm[i] + 1) + '; color ' + color_str)
            idx_done += [i]
            idx_done += [perm[i]]

    print('---')


def inv_perm(perm):

    inv_perm = np.empty(perm.size, perm.dtype)
    inv_perm[perm] = np.arange(perm.T.size)

    return inv_perm

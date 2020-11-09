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
    #adj_i = adj_set[i, :]
    v_i = v_set[i, :, :]

    #n_atoms = v_set.shape[1]
    #perm_ident = np.arange(n_atoms)


    #desc = Desc(n_atoms)





    match_perms = {}
    for j in range(i+1, n_train):

        adj_j = scipy.spatial.distance.squareform(adj_set[j, :])
        #adj_j = adj_set[j, :]
        v_j = v_set[j, :, :]

        cost = -np.fabs(v_i).dot(np.fabs(v_j).T)
        cost += same_z_cost * np.max(np.abs(cost))

        _, perm = scipy.optimize.linear_sum_assignment(cost)
        #tril_perm = desc.perm(perm)
        

        #if np.array_equiv(perm, perm_ident):
        #    print('sskip')

        #(A==B).all()

        adj_i_perm = adj_i[:, perm]
        adj_i_perm = adj_i_perm[perm, :]

        #print(adj_i.shape)
        #print(tril_perm.shape)


        #print(perm)

        #adj_i_perm = adj_i[tril_perm]

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

    match_cost = np.zeros((n_train, n_train))

    desc = Desc(n_atoms, max_processes=max_processes)

    adj_set = np.empty((n_train, desc.dim))
    v_set = np.empty((n_train, n_atoms, n_atoms))
    for i in range(n_train):
        r = np.squeeze(R[i, :, :])

        if lat_and_inv is None:
            adj = scipy.spatial.distance.pdist(r, 'euclidean')
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
        callback = partial(callback, disp_str='Multi-partite matching (permutation synchronization)')
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


def complete_sym_group(perms, callback=None):

    if callback is not None:
        callback = partial(callback, disp_str='Symmetry group completion')
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

    if callback is not None:
        callback(
            DONE,
            sec_disp_str='found {:d} symmetries'.format(perms.shape[0]),
        )

    return perms


def find_perms(R, z, lat_and_inv=None, callback=None, max_processes=None):

    #import timeit
    #start = timeit.default_timer()

    n_atoms = len(z)

    # find matching for all pairs
    match_perms_all, match_cost = bipartite_match(R, z, lat_and_inv, max_processes, callback=callback)

    # remove inconsistencies
    match_perms = sync_perm_mat(match_perms_all, match_cost, n_atoms, callback=callback)

    # commplete symmetric group
    sym_group_perms = complete_sym_group(match_perms, callback=callback)

    #stop = timeit.default_timer()
    #print((stop - start) / 2)

    return sym_group_perms


def find_frag_perms(R, z, lat_and_inv=None, callback=None, max_processes=None):

    from ase import Atoms
    from ase.build import molecule
    from ase.geometry.analysis import Analysis
    from scipy.sparse.csgraph import connected_components

    print('Finding permutable non-bonded fragments... (assumes Ang!)')

    # TODO: positions must be in Angstrom for this to work!!

    n_train, n_atoms = R.shape[:2]
    atoms = Atoms(z, positions=R[0]) # only use first molecule in dataset to find connected components (fix me later, maybe) # *0.529177249

    adj = Analysis(atoms).adjacency_matrix[0]
    _, labels = connected_components(csgraph=adj, directed=False, return_labels=True)

    frags = []
    for label in np.unique(labels):
        frags.append(np.where(labels == label)[0])
    n_frags = len(frags)

    if n_frags == n_atoms:
        print('Skipping fragment symmetry search (something went wrong, e.g. length unit not in Angstroms, etc.)')
        return [range(n_atoms)]

    #print(labels)





    #from . import ui, io
    #xyz_str = io.generate_xyz_str(R[0][np.where(labels == 0)[0], :]*0.529177249, z[np.where(labels == 0)[0]])
    #xyz_str = ui.indent_str(xyz_str, 2)
    #sprint(xyz_str)

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

    #print(n_frags)

    print('| Found ' + str(n_frags) + ' disconnected fragments.')

    # match fragments to find identical ones (allows permutations of fragments)
    swap_perms = [np.arange(n_atoms)]
    for f1 in range(n_frags):
        for f2 in range(f1+1,n_frags):

            #print(str(f1) + ' - ' + str(f2))

            sort_idx_f1 = np.argsort(z[frags[f1]])
            sort_idx_f2 = np.argsort(z[frags[f2]])
            inv_sort_idx_f2 = inv_perm(sort_idx_f2)

            for ri in range(min(10, R.shape[0])): # only use first molecule in dataset for matching (fix me later)

                R_match1 = R[ri,frags[f1], :]
                R_match2 = R[ri,frags[f2], :]
                z1 = z[frags[f1]][sort_idx_f1]
                z2 = z[frags[f2]][sort_idx_f2]

                if np.array_equal(z1, z2):
                    R_pair = np.concatenate((R_match1[None, sort_idx_f1, :], R_match2[None, sort_idx_f2, :]))

                    perms = find_perms(R_pair, z1, lat_and_inv=lat_and_inv, max_processes=max_processes)
                    for p in perms:

                        match_perm = sort_idx_f1[p][inv_sort_idx_f2]

                        swap_perm = np.arange(n_atoms)
                        swap_perm[frags[f1]] = frags[f2][match_perm]
                        swap_perm[frags[f2][match_perm]] = frags[f1]
                        swap_perms.append(swap_perm)

    swap_perms = np.unique(np.array(swap_perms), axis=0)

    #print('| Found ' + str(swap_perms.shape[0]) + ' unique fragment permutations.')

    # commplete symmetric group
    sym_group_perms = complete_sym_group(swap_perms)
    print('| Found ' + str(sym_group_perms.shape[0]) + ' fragment permutations after closure.')


    # match fragments with themselves (to find symmetries in each fragment)

    if n_frags > 1:
        print('| Matching individual fragments.')
        for f in range(n_frags):

            R_frag = R[:,frags[f],:]
            z_frag = z[frags[f]]

            #print(R_frag.shape)
            #print(z_frag)

            perms = find_perms(R_frag, z_frag, lat_and_inv=lat_and_inv, max_processes=max_processes)

            #print(f)
            print(perms)


    #sys.exit()


    #print('global perms')

    #r = R[0,:,:]

    #f1 = [364, 363, 362]
    #f2 = [362, 361, 360]

    #r1 = r[f1,:] - np.mean(r)
    #r2 = r[f2,:] - np.mean(r)

    #trans = np.linalg.solve(r1-np.mean(r), r2-np.mean(r))



    #H = r1.dot(r2)
    #u, s, vh = np.linalg.svd(H)
    #R = u.dot(vh)

    #print(R)



    #r_trans = (r-np.mean(r)).dot(R) + -np.mean(r)


    #stacked = np.vstack((r[None, ...], r_trans[None, ...]))

    #print(trans)

    


    #perms = find_perms(stacked, z, lat_and_inv=lat_and_inv, max_processes=max_processes)

    #print(perms.shape)

    #trans = np.dot(np.linalg.inv(r1), r2)




    #sys.exit()

    return sym_group_perms


# points as columns
def ralign(X,Y):

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1;
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1  
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t

    R = np.dot( np.dot(U, S ), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R,c,t


def inv_perm(perm):

    inv_perm = np.empty(perm.size, perm.dtype)
    inv_perm[perm] = np.arange(perm.size)

    return inv_perm

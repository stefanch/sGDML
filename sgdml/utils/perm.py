import sys
import scipy.spatial.distance
import scipy.optimize

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import numpy as np

import multiprocessing as mp
from functools import partial

import ui

glob = {}


def share_array(arr_np, typecode):
	arr = mp.RawArray(typecode, arr_np.ravel())
	return arr, arr_np.shape


def _sync_mat_wkr(i, n_train, same_z_cost):

	global glob

	adj_set = np.frombuffer(glob['adj_set']).reshape(glob['adj_set_shape'])
	v_set = np.frombuffer(glob['v_set']).reshape(glob['v_set_shape'])
	match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])

	adj_i = scipy.spatial.distance.squareform(adj_set[i,:])
	v_i = v_set[i,:,:]

	match_perms = {}
	for j in range(i+1,n_train):

		adj_j = scipy.spatial.distance.squareform(adj_set[j,:])
		v_j = v_set[j,:,:]

		cost = -np.fabs(v_i).dot(np.fabs(v_j).T)
		cost += same_z_cost * np.max(np.abs(cost))

		_,perm = scipy.optimize.linear_sum_assignment(cost)

		adj_i_perm = adj_i[:,perm]
		adj_i_perm = adj_i_perm[perm,:]

		score_before = np.linalg.norm(adj_i - adj_j)
		score = np.linalg.norm(adj_i_perm - adj_j)

		match_cost[i,j] = score
		if score >= score_before:
			match_cost[i,j] = score_before
		elif not np.isclose(score_before, score): # otherwise perm is identity
			match_perms[i,j] = perm

	return match_perms


def sync_mat(R,z,max_processes=None):

	global glob

	print ui.white_bold_str('Recovering symmetries...')

	n_train,n_atoms,_ = R.shape
	
	# penalty matrix for mixing atom species
	same_z_cost = np.repeat(z[:,None],len(z),axis=1) - z
	same_z_cost[same_z_cost != 0] = 1

	match_cost = np.zeros((n_train,n_train))

	adj_set = np.empty((n_train,(n_atoms**2 - n_atoms) / 2))
	v_set = np.empty((n_train,n_atoms,n_atoms))
	for i in range(n_train):
		r = np.squeeze(R[i,:,:])

		adj = scipy.spatial.distance.pdist(r,'euclidean')
		w,v = np.linalg.eig(scipy.spatial.distance.squareform(adj))
		v = v[:,w.argsort()[::-1]]

		adj_set[i,:] = adj
		v_set[i,:,:] = v

	glob['adj_set'], glob['adj_set_shape'] = share_array(adj_set, 'd')
	glob['v_set'], glob['v_set_shape'] = share_array(v_set, 'd')
	glob['match_cost'], glob['match_cost_shape'] = share_array(match_cost, 'd')

	pool = mp.Pool(max_processes)
	match_perms_all = {}
	for i,match_perms in enumerate(pool.imap_unordered(partial(_sync_mat_wkr, n_train=n_train, same_z_cost=same_z_cost), range(n_train))):			
		match_perms_all.update(match_perms)

		progr = float(i) / (n_train-1)
		sys.stdout.write('\r[%3d%%] Bi-partite matching...' % (progr * 100))
		sys.stdout.flush()
	print ''
	pool.close()

	match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])
	match_cost = match_cost + match_cost.T
	match_cost[np.diag_indices_from(match_cost)] = np.inf
	match_cost = csr_matrix(match_cost)

	tree = minimum_spanning_tree(match_cost, overwrite=True)

	perms = np.arange(n_atoms, dtype=int)[None, :]
	rows,cols = tree.nonzero()
	for com in zip(rows,cols):
		perm = match_perms_all.get(com)
		if perm is not None:
			perms = np.vstack((perms, perm))
	perms = np.unique(perms, axis=0)
	sys.stdout.write('[DONE] Permutation synchronization...')
	sys.stdout.flush()

	return perms


def complete_group(perms):

	perm_added = True
	while perm_added:
		perm_added = False

		n_perms = perms.shape[0]
		for i in range(n_perms):
			for j in range(n_perms):

				new_perm = perms[i,perms[j,:]]
				if not (new_perm == perms).all(axis=1).any():
					perm_added = True
					perms = np.vstack((perms, new_perm))

	print ui.gray_str(' (%d symmetries)' % perms.shape[0])
	return perms


def inv_perm(perm):

    inv_perm = np.empty(perm.size, perm.dtype)
    inv_perm[perm] = np.arange(perm.size)

    return inv_perm


def to_tril_perm(perm):

	n = len(perm)
	perm = perm # - 1 # MATLAB is 1-dominant (legacy reasons)

	rest = np.zeros((n,n))
	rest[np.tril_indices(n,-1)] = range(0,(n**2-n)/2)
	rest = rest + rest.T
	rest = rest[perm, :]
	rest = rest[:, perm]
	
	return rest[np.tril_indices(n,-1)].astype(int)

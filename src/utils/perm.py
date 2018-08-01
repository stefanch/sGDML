import sys
import scipy.spatial.distance
import scipy.optimize

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import numpy as np

import multiprocessing as mp
from functools import partial

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
		cost -= same_z_cost * np.max(cost)

		_,perm = scipy.optimize.linear_sum_assignment(cost)

		adj_i_perm = adj_i[:,perm]
		adj_i_perm = adj_i_perm[perm,:]

		score_before = np.linalg.norm(adj_i - adj_j)
		score = np.linalg.norm(adj_i_perm - adj_j)

		match_cost[i,j] = score
		if score - score_before > 0:
			match_cost[i,j] = score_before
		elif not np.isclose(score_before, score): # otherwise perm is identity
			match_perms[i,j] = perm

		#no += 1
		#progr = float(no) / ((n_train**2 - n_train) / 2)
		#sys.stdout.write('\r \x1b[1;37m[%3d%%]\x1b[0m Bi-partite matching...' % (progr * 100))
		#sys.stdout.flush()

	return match_perms


def sync_mat(R,z):

	global glob

	print 'Recovering symmetries...'

	n_train,n_atoms,_ = R.shape
	
	# penalty matrix for mixing atom species
	same_z_cost = np.repeat(z[:,None],len(z),axis=1) - z
	same_z_cost[same_z_cost != 0] = 1

	match_cost = np.zeros((n_train,n_train))
	#match_perms = {}

	adj_set = np.empty((n_train,(n_atoms**2 - n_atoms) / 2))
	v_set = np.empty((n_train,n_atoms,n_atoms))
	for i in range(n_train):
		r = np.squeeze(R[i,:,:])
		adj = scipy.spatial.distance.pdist(r,'euclidean')
		_,v = np.linalg.eig(scipy.spatial.distance.squareform(adj))

		adj_set[i,:] = adj
		v_set[i,:,:] = v


	#glob['R'], glob['R_shape'] = share_array(R, 'd')
	glob['adj_set'], glob['adj_set_shape'] = share_array(adj_set, 'd')
	glob['v_set'], glob['v_set_shape'] = share_array(v_set, 'd')
	glob['match_cost'], glob['match_cost_shape'] = share_array(match_cost, 'd')

	# no = 0
	# for i,r_i in enumerate(R):
		
	# 	adj_i = scipy.spatial.distance.squareform(adj_set[i,:])
	# 	v_i = v_set[i,:,:]

	# 	for j in range(i+1,n_train):

	# 		adj_j = scipy.spatial.distance.squareform(adj_set[j,:])
	# 		v_j = v_set[j,:,:]

	# 		cost = -np.fabs(v_i).dot(np.fabs(v_j).T)
	# 		cost -= same_z_cost * np.max(cost)

	# 		_,perm = scipy.optimize.linear_sum_assignment(cost)

	# 		adj_i_perm = adj_i[:,perm]
	# 		adj_i_perm = adj_i_perm[perm,:]

	# 		score_before = np.linalg.norm(adj_i - adj_j)
	# 		score = np.linalg.norm(adj_i_perm - adj_j)

	# 		match_cost[i,j] = score
	# 		if score - score_before > 0:
	# 			match_cost[i,j] = score_before
	# 		elif not np.isclose(score_before, score): # otherwise perm is identity
	# 			match_perms[i,j] = perm

	# 		no += 1
	# 		progr = float(no) / ((n_train**2 - n_train) / 2)
	# 		sys.stdout.write('\r \x1b[1;37m[%3d%%]\x1b[0m Bi-partite matching...' % (progr * 100))
	# 		sys.stdout.flush()
	# print ''

	pool = mp.Pool()
	match_perms_all = {}
	for i,match_perms in enumerate(pool.imap_unordered(partial(_sync_mat_wkr, n_train=n_train, same_z_cost=same_z_cost), range(n_train))):			
		match_perms_all.update(match_perms)

		progr = float(i) / (n_train-1)
		sys.stdout.write('\r \x1b[1;37m[%3d%%]\x1b[0m Bi-partite matching...' % (progr * 100))
		sys.stdout.flush()
	print ''
	pool.close()

	match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])
	match_cost = match_cost + match_cost.T
	match_cost[np.diag_indices_from(match_cost)] = np.inf
	match_cost = csr_matrix(match_cost)

	tree = minimum_spanning_tree(match_cost, overwrite=True)
	#_,pred = scipy.sparse.csgraph.shortest_path(tree, directed=False, return_predecessors=True) # first argument: dist matrix

	#perms = np.empty((0,n_atoms),dtype=int)
	perms = np.arange(n_atoms, dtype=int)[None, :]
	rows,cols = tree.nonzero()
	for com in zip(rows,cols):
		#perm = match_perms[com[0]].get(com)
		perm = match_perms_all.get(com)
		#perm = match_perms.get(com)
		if perm is not None:
			perms = np.vstack((perms, perm))
	perms = np.unique(perms, axis=0)
	sys.stdout.write(' \x1b[1;37m[DONE]\x1b[0m Multi-partite matching...')
	sys.stdout.flush()

	return perms







# def sync_mat(R,z):

# 	return sync_mat_multi(R,z)

# 	print 'Recovering symmetries...'

# 	n_train,n_atoms,_ = R.shape
	
# 	# penalty matrix for mixing atom species
# 	same_z_cost = np.repeat(z[:,None],len(z),axis=1) - z
# 	same_z_cost[same_z_cost != 0] = 1

# 	match_cost = np.zeros((n_train,n_train))
# 	match_perms = {}

# 	adj_set = np.empty((n_train,(n_atoms**2 - n_atoms) / 2))
# 	v_set = np.empty((n_train,n_atoms,n_atoms))
# 	for i in range(n_train):
# 		r = np.squeeze(R[i,:,:])
# 		adj = scipy.spatial.distance.pdist(r,'euclidean')
# 		_,v = np.linalg.eig(scipy.spatial.distance.squareform(adj))

# 		adj_set[i,:] = adj
# 		v_set[i,:,:] = v

# 	no = 0
# 	for i,r_i in enumerate(R):
		
# 		adj_i = scipy.spatial.distance.squareform(adj_set[i,:])
# 		v_i = v_set[i,:,:]

# 		for j in range(i+1,n_train):

# 			adj_j = scipy.spatial.distance.squareform(adj_set[j,:])
# 			v_j = v_set[j,:,:]

# 			cost = -np.fabs(v_i).dot(np.fabs(v_j).T)
# 			cost -= same_z_cost * np.max(cost)

# 			_,perm = scipy.optimize.linear_sum_assignment(cost)

# 			adj_i_perm = adj_i[:,perm]
# 			adj_i_perm = adj_i_perm[perm,:]

# 			score_before = np.linalg.norm(adj_i - adj_j)
# 			score = np.linalg.norm(adj_i_perm - adj_j)

# 			match_cost[i,j] = score
# 			if score - score_before > 0:
# 				match_cost[i,j] = score_before
# 			elif not np.isclose(score_before, score): # otherwise perm is identity
# 				match_perms[i,j] = perm

# 			no += 1
# 			progr = float(no) / ((n_train**2 - n_train) / 2)
# 			sys.stdout.write('\r \x1b[1;37m[%3d%%]\x1b[0m Bi-partite matching...' % (progr * 100))
# 			sys.stdout.flush()
# 	print ''

# 	match_cost = match_cost + match_cost.T
# 	match_cost[np.diag_indices_from(match_cost)] = np.inf
# 	match_cost = csr_matrix(match_cost)


# 	tree = minimum_spanning_tree(match_cost, overwrite=True)
# 	#_,pred = scipy.sparse.csgraph.shortest_path(tree, directed=False, return_predecessors=True) # first argument: dist matrix

# 	perms = np.empty((0,n_atoms),dtype=int)
# 	rows,cols = tree.nonzero()
# 	for com in zip(rows,cols):
# 		perm = match_perms.get(com)
# 		if perm is not None:
# 			perms = np.vstack((perms, perm))
# 	perms = np.unique(perms, axis=0)
# 	print ' \x1b[1;37m[DONE]\x1b[0m Multi-partite matching...'

# 	return perms

# def get_perm(s,t,pred,match_perms,n_atoms): # from_min_span_tree

# 	perm_st = np.arange(n_atoms, dtype=np.uint16)

# 	if s == t:
# 		return perm_st

# 	prev_t = t
# 	t = pred[s,t]
# 	while t >= 0:

# 		new_p = match_perms.get((min(t,prev_t),max(t,prev_t)))
# 		if new_p is not None:
# 			if prev_t > t:
# 				new_p = inv_perm(new_p)
# 			perm_st = perm_st[new_p]

# 		prev_t = t
# 		t = pred[s,t]

# 	return perm_st

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

	print ' (Found ' + str(perms.shape[0]) + '.)'
	return perms

def inv_perm(perm):

    inv_perm = np.empty(perm.size, perm.dtype)
    inv_perm[perm] = np.arange(perm.size)

    return inv_perm

def to_tril_perm(perm):

	n = len(perm)
	perm = perm# - 1 # MATLAB is 1-dominant (legacy reasons)

	rest = np.zeros((n,n))
	rest[np.tril_indices(n,-1)] = range(0,(n**2-n)/2)
	rest = rest + rest.T
	rest = rest[perm, :]
	rest = rest[:, perm]
	
	return rest[np.tril_indices(n,-1)].astype(int)

import sys
import scipy.spatial.distance
import scipy.optimize

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import numpy as np

import multiprocessing as mp
from functools import partial

def sync_mat(R,z):

	print 'Recovering symmetries...'

	n_train,n_atoms,_ = R.shape
	
	# penalty matrix for mixing atom species
	same_z_cost = np.repeat(z[:,None],len(z),axis=1) - z
	same_z_cost[same_z_cost != 0] = 1

	match_cost = np.zeros([n_train,n_train])
	match_perms = {}

	adj_set = np.empty((n_train,(n_atoms**2 - n_atoms) / 2))
	v_set = np.empty((n_train,n_atoms,n_atoms))
	for i in range(0,n_train):
		r = np.squeeze(R[i,:,:])
		adj = scipy.spatial.distance.pdist(r,'euclidean')
		_,v = np.linalg.eig(scipy.spatial.distance.squareform(adj))

		adj_set[i,:] = adj
		v_set[i,:,:] = v

	no = 0
	for i,r_i in enumerate(R):
		
		adj_i = scipy.spatial.distance.squareform(adj_set[i,:])
		v_i = v_set[i,:,:]

		for j in range(i+1,n_train):

			adj_j = scipy.spatial.distance.squareform(adj_set[j,:])
			v_j = v_set[j,:,:]

			cost = -np.fabs(v_i).dot(np.fabs(v_j).T)
			cost -= same_z_cost * np.max(cost)

			_,perm = scipy.optimize.linear_sum_assignment(cost)

			adj_i_perm = adj_i[:,perm]
			adj_i_perm = adj_i_perm[perm,:]

			score = np.linalg.norm(adj_i_perm - adj_j)
			score_before = np.linalg.norm(adj_i - adj_j)

			match_cost[i,j] = score
			if score - score_before > 0:
				match_cost[i,j] = score_before
			else:
				match_perms[i,j] = perm

			no += 1
			progr = float(no) / ((n_train**2 - n_train) / 2)
			sys.stdout.write('\r \x1b[1;37m[%3d%%]\x1b[0m Bi-partite matching...' % (progr * 100))
			sys.stdout.flush()
	print ''

	match_cost = match_cost + match_cost.T
	match_cost[np.diag_indices_from(match_cost)] = np.inf

	match_cost = csr_matrix(match_cost)

	tree = minimum_spanning_tree(match_cost, overwrite=True)
	_,pred = scipy.sparse.csgraph.shortest_path(tree, directed=False, return_predecessors=True) # first argument: dist matrix

	# extract column 1
	#pool = mp.Pool()
	#perms = np.array(pool.map(partial(get_perm, t=1, pred=pred, match_perms=match_perms, n_atoms=n_atoms), range(n_train)))
	#col_perms = np.unique(perms, axis=0)

	# extract column 1
	pool = mp.Pool()
	perms = np.empty((0,n_atoms),dtype=int)
	for i,perm in enumerate(pool.imap_unordered(partial(get_perm, t=1, pred=pred, match_perms=match_perms, n_atoms=n_atoms), range(n_train))):			

		perms = np.vstack((perms, perm))
		perms = np.unique(perms, axis=0)

	 	progr = float(i+1) / n_train
	 	sys.stdout.write("\r \x1b[1;37m[%3d%%]\x1b[0m Transitive closure..." % (progr * 100))
	 	sys.stdout.flush()
	pool.close()

	return perms

def get_perm(s,t,pred,match_perms,n_atoms): # from_min_span_tree

	perm_st = np.arange(n_atoms, dtype=np.uint16)

	prev_t = t
	t = pred[s,t]
	while t >= 0:
		if (t,prev_t) in match_perms.keys():
			perm_st = perm_st[match_perms[t,prev_t]]
		elif (prev_t,t) in match_perms.keys():
			perm_st = perm_st[inv_perm(match_perms[prev_t,t])]

		prev_t = t
		t = pred[s,t]

	return perm_st

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
					perms = np.vstack([perms, new_perm])

	print ' (Found ' + str(perms.shape[0]) + '.)'
	return perms

def inv_perm(perm):

    inv_perm = np.empty(perm.size, perm.dtype)
    inv_perm[perm] = np.arange(perm.size)

    return inv_perm

def to_tril_perm(perm):

	n = len(perm)
	perm = perm - 1 # MATLAB is 1-dominant (legacy reasons)

	rest = np.zeros((n,n))
	rest[np.tril_indices(n,-1)] = range(0,(n**2-n)/2)
	rest = rest + rest.T
	rest = rest[perm, :]
	rest = rest[:, perm]
	
	return rest[np.tril_indices(n,-1)].astype(int)

import numpy as np

d_desc_mask = None

def init(n_atoms):
	global d_desc_mask
	
	# Precompute indices for nonzero entries in desriptor derivatices.
	d_desc_mask = np.zeros((n_atoms,n_atoms-1), dtype=np.int)
	for a in range(0,n_atoms): # for each partial deriavative
		rows,cols = np.tril_indices(n_atoms,-1)
		d_desc_mask[a,:] = np.concatenate([np.where( rows == a)[0], np.where( cols == a)[0]])

def r_to_desc(r,pdist):
	n_atoms = r.shape[0]
	return 1. / pdist[np.tril_indices(n_atoms,-1)]

def r_to_d_desc(r,pdist):
	global d_desc_mask

	n_atoms = r.shape[0]
	d_dim = (n_atoms**2 - n_atoms)/2

	if d_desc_mask is None:
		init(n_atoms)

	np.seterr(divide='ignore', invalid='ignore') # ignore division by zero below
	grad = np.zeros((d_dim,3*n_atoms))
	for a in range(0,n_atoms):

		d_dist = (r - r[a,:]) / (pdist[a,:]**3)[:,None]

		idx = d_desc_mask[a,:]
		grad[idx,(3*a):(3*a+3)] = np.delete(d_dist, a, axis=0)

	return grad
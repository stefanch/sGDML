import numpy as np

d_desc_mask = None

def init(n_atoms):
	global d_dim, d_desc_mask

	# Descriptor space dimension.
	d_dim = (n_atoms * (n_atoms-1)) / 2
	
	# Precompute indices for nonzero entries in desriptor derivatices.
	d_desc_mask = np.zeros((n_atoms,n_atoms-1), dtype=np.int)
	for a in range(n_atoms): # for each partial deriavative
		rows,cols = np.tril_indices(n_atoms,-1)
		d_desc_mask[a,:] = np.concatenate([np.where(rows == a)[0], np.where(cols == a)[0]])

#def desc_dim(r):
#	n_atoms = r.shape[0]
#	d_dim = (n_atoms**2 - n_atoms)/2

def r_to_desc(r,pdist):
	n_atoms = r.shape[0]
	return 1. / pdist[np.tril_indices(n_atoms,-1)]

def r_to_d_desc(r,pdist):
	global d_dim, d_desc_mask

	n_atoms = r.shape[0]

	if d_desc_mask is None:
		init(n_atoms)

	np.seterr(divide='ignore', invalid='ignore') # ignore division by zero below
	grad = np.zeros((d_dim,3*n_atoms))
	for a in range(n_atoms):

		d_dist = (r - r[a,:]) / (pdist[a,:]**3)[:,None]

		idx = d_desc_mask[a,:]
		grad[idx,(3*a):(3*a+3)] = np.delete(d_dist, a, axis=0)

	return grad

def r_to_d_desc_op(r,pdist,F_d): # returns F_d.dot(r_d_desc)
	global d_dim, d_desc_mask

	n_atoms = r.shape[0]
	
	if d_desc_mask is None:
		init(n_atoms)

	np.seterr(divide='ignore', invalid='ignore') # ignore division by zero below
	F_i = np.empty((3*n_atoms,))
	for a in range(n_atoms):

		d_dist = (r - r[a,:]) / (pdist[a,:]**3)[:,None]

		idx = d_desc_mask[a,:]
		F_d[idx].dot(np.delete(d_dist, a, axis=0), out=F_i[(3*a):(3*a+3)])

	return F_i

# converts to permutation in desc space
def perm(perm):

	n = len(perm)
	#perm = perm # - 1 # MATLAB is 1-dominant (legacy reasons)

	rest = np.zeros((n,n))
	rest[np.tril_indices(n,-1)] = range((n**2-n)/2)
	rest = rest + rest.T
	rest = rest[perm, :]
	rest = rest[:, perm]
	
	return rest[np.tril_indices(n,-1)].astype(int)
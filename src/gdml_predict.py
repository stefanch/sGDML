# GDML Force Field
# Author: Stefan Chmiela (stefan@chmiela.com)
VERSION = 180702

import sys

import scipy.spatial.distance
import numpy as np

import multiprocessing as mp
from functools import partial

import config

from utils import desc


glob = {}

def share_array(arr_np):
	arr = mp.RawArray('d', arr_np.ravel())
	return arr, arr_np.shape

def predict_worker_cached(wkr_start_stop, r_desc, r_d_desc):

	global glob,sig,n_perms,b_size
	R_desc_perms = np.frombuffer(glob['R_desc_perms']).reshape(glob['R_desc_perms_shape'])
	R_d_desc_alpha_perms = np.frombuffer(glob['R_d_desc_alpha_perms']).reshape(glob['R_d_desc_alpha_perms_shape'])

	wkr_start, wkr_stop = wkr_start_stop

	mat52_base_fact = 5./(3.*sig**3)
	diag_scale_fact = 5./sig
	sqrt5 = np.sqrt(5.)

	E = 0.
	F = np.zeros((r_d_desc.shape[1],))

	wkr_start *= n_perms
	wkr_stop *= n_perms

	b_start = wkr_start
	for b_stop in range(wkr_start+b_size*n_perms,wkr_stop,b_size*n_perms) + [wkr_stop]:

		rj_desc_perms = R_desc_perms[b_start:b_stop,:]
		rj_d_desc_alpha_perms = R_d_desc_alpha_perms[b_start:b_stop,:]

		diff_ab_perms = r_desc - rj_desc_perms
		norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

		mat52_base = np.exp(-norm_ab_perms / sig) * mat52_base_fact
		a_x2 = np.einsum('ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms) # colum wise dot product

		#F += np.einsum('ji,j', diff_ab_perms.dot(r_d_desc), a_x2) * diag_scale_fact # correct
		F += np.linalg.multi_dot([a_x2 * mat52_base, diff_ab_perms, r_d_desc]) * diag_scale_fact
		mat52_base *= norm_ab_perms + sig

		#F += np.einsum('ji,j', rj_d_desc_alpha_perms.dot(r_d_desc), -mat52_base) #correct
		F -= np.linalg.multi_dot([mat52_base, rj_d_desc_alpha_perms, r_d_desc])
		E += a_x2.dot(mat52_base)

		b_start = b_stop

	return np.append(F, E)


class GDMLPredict:

	def __init__(self,model,batch_size=250,num_workers=None):

		global glob,perms_lin,sig,n_perms,b_size

		# Batch size (number of training samples summed up in prediction process) that a worker processes at once.
		b_size = batch_size

		self.n_atoms = model['z'].shape[0]
		n_tril = (self.n_atoms**2 - self.n_atoms) / 2
		
		n_train = model['R_desc'].shape[1]
		sig = model['sig']
		self.c = model['c']

		# Precompute permutated training descriptors and its first derivatives multiplied with the coefficients (only needed for cached variant).
		n_perms = model['tril_perms_lin'].shape[0] / n_tril
		R_desc_perms = np.reshape(np.tile(model['R_desc'].T, n_perms)[:,model['tril_perms_lin']], (n_train*n_perms,-1), order='F')
		glob['R_desc_perms'], glob['R_desc_perms_shape'] = share_array(R_desc_perms)
		R_d_desc_alpha_perms = np.reshape(np.tile(np.squeeze(model['R_d_desc_alpha']), n_perms)[:,model['tril_perms_lin']], (n_train*n_perms,-1), order='F')
		glob['R_d_desc_alpha_perms'], glob['R_d_desc_alpha_perms_shape'] = share_array(R_d_desc_alpha_perms)

		self.n_procs = num_workers
		self.pool = mp.Pool(processes=self.n_procs)

		# Data ranges for processes
		wkr_starts = range(0,n_train,int(np.ceil(float(n_train)/self.pool._processes)))
		wkr_stops = wkr_starts[1:] + [n_train]
		self.wkr_starts_stops = zip(wkr_starts,wkr_stops)

	def __del__(self):
		self.pool.close()

	def version(self):
		return VERSION


	## Public ##

	def _predict_bulk(self,R):

		n_pred, dim_i = R.shape

		F = np.empty((n_pred,dim_i))
		E = np.empty((n_pred,))
		for i,r in enumerate(R):
			E[i],F[i,:] = self.predict(r)

		return (E.ravel(), F.ravel())

	# input:  r [M,N*3] -> [[x11,y11,z11, ..., x1N,y1N,z1N], ..., [xM1,yM1,zM1, ..., xMN,yMN,zMN]]
	# return: F [M,N*3] -> [[x11,y11,z11, ..., x1N,y1N,z1N], ..., [xM1,yM1,zM1, ..., xMN,yMN,zMN]]
	def predict(self,r):

		if r.ndim == 2 and r.shape[0] > 1:
			return self._predict_bulk(r)

		r = r.reshape(self.n_atoms,3)
		pdist = scipy.spatial.distance.pdist(r,'euclidean')
		pdist = scipy.spatial.distance.squareform(pdist)

		r_desc = desc.r_to_desc(r,pdist)
		r_d_desc = desc.r_to_d_desc(r,pdist)

		res = sum(self.pool.map(partial(predict_worker_cached, r_desc=r_desc, r_d_desc=r_d_desc), self.wkr_starts_stops))
		return (res[-1]+self.c, res[:-1].reshape(1,-1))

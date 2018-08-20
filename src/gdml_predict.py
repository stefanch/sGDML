# GDML Force Field
# Author: Stefan Chmiela (stefan@chmiela.com)
VERSION = 310702

import sys

import scipy.spatial.distance
import numpy as np

import time
import timeit
import multiprocessing as mp
from functools import partial

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

		global glob,perms_lin,sig,n_perms

		# Batch size (number of training samples summed up in prediction process) that a worker processes at once.
		self.set_batch_size(batch_size)

		self.n_atoms = model['z'].shape[0]
		n_tril = (self.n_atoms**2 - self.n_atoms) / 2
		
		self.n_train = model['R_desc'].shape[1]
		sig = model['sig']
		self.c = model['c']

		# Precompute permutated training descriptors and its first derivatives multiplied with the coefficients (only needed for cached variant).
		n_perms = model['tril_perms_lin'].shape[0] / n_tril
		
		R_desc_perms = np.reshape(np.tile(model['R_desc'].T, n_perms)[:,model['tril_perms_lin']], (self.n_train*n_perms,-1), order='F')
		glob['R_desc_perms'], glob['R_desc_perms_shape'] = share_array(R_desc_perms)
		
		R_d_desc_alpha_perms = np.reshape(np.tile(np.squeeze(model['R_d_desc_alpha']), n_perms)[:,model['tril_perms_lin']], (self.n_train*n_perms,-1), order='F')
		glob['R_d_desc_alpha_perms'], glob['R_d_desc_alpha_perms_shape'] = share_array(R_d_desc_alpha_perms)

		self.pool = None
		self.set_num_workers(num_workers)

	def __del__(self):
		self.pool.close()

	def version(self):
		return VERSION

	## Public ##

	def set_num_workers(self, num_workers):
		if self.pool is not None:
			self.pool.close()
		self.pool = mp.Pool(processes=num_workers)

		# Data ranges for processes
		wkr_starts = range(0,self.n_train,int(np.ceil(float(self.n_train)/self.pool._processes)))
		wkr_stops = wkr_starts[1:] + [self.n_train]
		self.wkr_starts_stops = zip(wkr_starts,wkr_stops)

		self._num_workers = num_workers

	def set_batch_size(self, batch_size):

		global b_size

		b_size = batch_size
		self._batch_size = batch_size

	# sets best (num_workers, batch_size) for model
	# the model is either tested on a dummy input or geometries from a provided set
	# note: the optimal parameters are NOT set if this function runs out of geometries (if provided)
	# return_when_R_done: returns as soons as all geos in R are used up, even before n_reps is done
	# def set_opt_num_workers_and_batch_size(self, R=None, n_reps=100, return_when_R_done=False):

	# 	F = np.empty((0,self.n_atoms*3))
	# 	E = []

	# 	best_sps = 0
	# 	best_params = 1,1
	# 	best_results = []
	# 	r_dummy = np.random.rand(1,self.n_atoms*3)

	# 	#import timeit
	# 	#def _dummy_predict():
	# 	#	self.predict(r_dummy)

	# 	for num_workers in range(1,mp.cpu_count()+1):
	# 		if self.n_train % num_workers != 0:
	# 			continue
	# 		self.set_num_workers(num_workers)

	# 		best_sps = 0
	# 		for batch_size in range(int(np.ceil(self.n_train/num_workers)), 0, -1):
	# 			if self.n_train % batch_size != 0:
	# 				continue
	# 			self.set_batch_size(batch_size)

	# 			#print 1. / (timeit.timeit(_dummy_predict, number=100) / 100.)


	# 			t_elap = 0
	# 			for i in range(1,n_reps+1):
	# 				R_done = R is None or len(E) == R.shape[0]
	# 				if R_done:
	# 					t = time.time()
	# 					self.predict(r_dummy)

						
						

	# 					t_elap += time.time() - t
	# 				else:
	# 					r = R[len(E)].reshape(1,-1)

	# 					t = time.time()
	# 					e,f = self.predict(r)
	# 					t_elap += time.time() - t

	# 					E.append(e)
	# 					F = np.vstack((F,f))

	# 					# finished computing geos in R
	# 					if len(E) == R.shape[0] and return_when_R_done:
	# 						return np.array(E),F.ravel()

	# 				if t_elap > 1: # don't spend more than 1s on this
	# 					break
	# 			sps = i / t_elap

	# 			if sps < best_sps:
	# 				break
	# 			else:
	# 				best_sps = sps
	# 				best_params = num_workers, batch_size

	# 			print '{:2d}@{:d} | {:7.2f} sps'.format(num_workers,batch_size, sps)
	# 		best_results.append((best_params, best_sps))
		
	# 	num_workers, batch_size = max(best_results, key=lambda x:x[1])[0]
				
	# 	self.set_batch_size(batch_size)
	# 	self.set_num_workers(num_workers)

	# 	return np.array(E),F.ravel()


	# sets best level of parallelism (num_workers, batch_size) for model
	def set_opt_num_workers_and_batch_size_fast(self, n_reps=100):

		best_sps = 0
		best_params = 1,1
		best_results = []
		r_dummy = np.random.rand(1,self.n_atoms*3)

		def _dummy_predict():
			self.predict(r_dummy)

		for num_workers in range(1,mp.cpu_count()+1):
			if self.n_train % num_workers != 0:
				continue
			self.set_num_workers(num_workers)

			best_sps = 0
			for batch_size in range(int(np.ceil(self.n_train/num_workers)), 0, -1):
				if self.n_train % batch_size != 0:
					continue
				self.set_batch_size(batch_size)
				
				sps = n_reps / (timeit.timeit(_dummy_predict, number=n_reps))
				if sps < best_sps:
					break
				else:
					best_sps = sps
					best_params = num_workers, batch_size

			#print '{:2d}@{:d} | {:7.2f} sps'.format(num_workers,batch_size, sps)
			if len(best_results) > 0 and best_sps < best_results[-1][1]:
				break

			best_results.append((best_params, best_sps))
		
		num_workers, batch_size = max(best_results, key=lambda x:x[1])[0]
				
		self.set_batch_size(batch_size)
		self.set_num_workers(num_workers)

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

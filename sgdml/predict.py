"""
This module contains all routines for evaluating GDML and sGDML models.
"""

# MIT License
# 
# Copyright (c) 2018 Stefan Chmiela
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

#VERSION = 310702

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
	"""
	Return a ctypes array allocated from shared memory with data from
	a NumPy array of type `float`.

	Parameters
	----------
		arr_np : :obj:`numpy.ndarray`
			NumPy array.

	Returns
	-------
		array of :obj:`ctype`
	"""

	arr = mp.RawArray('d', arr_np.ravel())
	return arr, arr_np.shape

def predict_worker_cached(wkr_start_stop, r_desc, r_d_desc):
	"""
	Compute part of a prediction.

	The workload will be processed in `b_size` chunks.

	Parameters
	----------
		wkr_start_stop : tuple of int
			Indices of first and last (exclusive) sum element.
		r_desc : numpy.ndarray
			1D array containing the descriptor for the query geometry.
		r_d_desc : numpy.ndarray
			2D array containing the gradient of the descriptor for the
			query geometry.

	Returns
	-------
		:obj:`numpy.ndarray`
			Partial prediction of all force components and energy
			(appended to array as last element).
	"""

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

		F += np.linalg.multi_dot([a_x2 * mat52_base, diff_ab_perms, r_d_desc]) * diag_scale_fact
		mat52_base *= norm_ab_perms + sig

		F -= np.linalg.multi_dot([mat52_base, rj_d_desc_alpha_perms, r_d_desc])
		E += a_x2.dot(mat52_base)

		b_start = b_stop

	return np.append(F, E)


class GDMLPredict:

	def __init__(self, model, batch_size=250, num_workers=None):

		global glob,perms_lin,sig,n_perms

		# Batch size (number of training samples summed up in prediction process) that a worker processes at once.
		self.set_batch_size(batch_size)

		self.n_atoms = model['z'].shape[0]
		n_tril = (self.n_atoms**2 - self.n_atoms) / 2
		
		self.n_train = model['R_desc'].shape[1]
		sig = model['sig']
		self.c = model['c']

		# Precompute permuted training descriptors and its first derivatives multiplied with the coefficients (only needed for cached variant).
		n_perms = model['tril_perms_lin'].shape[0] / n_tril
		
		R_desc_perms = np.reshape(np.tile(model['R_desc'].T, n_perms)[:,model['tril_perms_lin']], (self.n_train*n_perms,-1), order='F')
		glob['R_desc_perms'], glob['R_desc_perms_shape'] = share_array(R_desc_perms)
		
		R_d_desc_alpha_perms = np.reshape(np.tile(np.squeeze(model['R_d_desc_alpha']), n_perms)[:,model['tril_perms_lin']], (self.n_train*n_perms,-1), order='F')
		glob['R_d_desc_alpha_perms'], glob['R_d_desc_alpha_perms_shape'] = share_array(R_d_desc_alpha_perms)

		self.pool = None
		self.set_num_workers(num_workers)

	def __del__(self):
		self.pool.close()

	#def version(self):
	#	return VERSION


	## Public ##

	def set_num_workers(self, num_workers):
		"""
		Set number of processes to use during prediction.

		This number should not exceed the number of available CPU cores.

		Note
		----
			This parameter can be optimally determined using
			`set_opt_num_workers_and_batch_size_fast`.

		Parameters
		----------
			num_workers : int
				Number of processes.
		"""

		if self.pool is not None:
			self.pool.close()
		self.pool = mp.Pool(processes=num_workers)

		# Data ranges for processes
		wkr_starts = range(0, self.n_train,int(np.ceil(float(self.n_train)/self.pool._processes)))
		wkr_stops = wkr_starts[1:] + [self.n_train]
		self.wkr_starts_stops = zip(wkr_starts, wkr_stops)

		self._num_workers = num_workers

	def set_batch_size(self, batch_size):
		"""
		Set chunk size for each process.

		The chunk size determines how much of a processes workload
		will be passed to Python's underlying low-level routines at
		once. This parameter is highly hardware dependent.
		A chunk is a subset of the training set of the model.

		Note
		----
			This parameter can be optimally determined using
			`set_opt_num_workers_and_batch_size_fast`.

		Parameters
		----------
			batch_size : int
				Chunk size.
		"""

		global b_size

		b_size = batch_size
		self._batch_size = batch_size

	def set_opt_num_workers_and_batch_size_fast(self, n_reps=100):
		"""
		Determine the optimal number of processes and chunk size to
		use when evaluating the loaded model.

		This routine runs a benchmark in which the prediction routine
		in repeatedly called `n_reps`-times with varying parameter
		configurations, while the runtime is measured for each one.
		The optimal parameters are then automatically set.

		Note
		----
			Depending on the parameter `n_reps`, this routine takes
			some seconds to complete, which is why it only makes sense
			to call it before running a large number of predictions.

		Parameters
		----------
			n_reps : int
				Number of repetitions (bigger value: more accurate,
				but also slower).
		"""

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
			for batch_size in range(int(np.ceil(self.n_train/num_workers))+1, 0, -1):
				if self.n_train % batch_size != 0:
					continue
				self.set_batch_size(batch_size)
				
				sps = n_reps / (timeit.timeit(_dummy_predict, number=n_reps))
				if sps < best_sps:
					break
				else:
					best_sps = sps
					best_params = num_workers, batch_size

				#print '{:2d}@{:d} | {:7.2f} sps'.format(num_workers, batch_size, sps)
			if len(best_results) > 0 and best_sps < best_results[-1][1]:
				break

			best_results.append((best_params, best_sps))
		
		num_workers, batch_size = max(best_results, key=lambda x:x[1])[0]
				
		self.set_batch_size(batch_size)
		self.set_num_workers(num_workers)

	def _predict_bulk(self,R):
		"""
		Predict energy and forces for multiple geometries.

		Parameters
		----------
			R : :obj:`numpy.ndarray`
				A 2D array of size M x 3N containing of the Cartesian coordinates of each
				atom of M molecules.

		Returns
		-------
			:obj:`numpy.ndarray`
				Energies stored in an 1D array of size M.
			:obj:`numpy.ndarray`
				Forces stored in an 2D arry of size M x 3N.
		"""

		n_pred, dim_i = R.shape

		F = np.empty((n_pred, dim_i))
		E = np.empty((n_pred,))
		for i,r in enumerate(R):
			E[i],F[i,:] = self.predict(r)

		return E, F

	def predict(self,r):
		"""
		Predict energy and forces for multiple geometries.

		Note
		----
			The order of the atoms in `r` is not arbitrary and must be
			the same as used for training the model.

		Parameters
		----------
			R : :obj:`numpy.ndarray`
				A 2D array of size M x 3N containing of the Cartesian coordinates of each
				atom of M molecules.

		Returns
		-------
			:obj:`numpy.ndarray`
				Energies stored in an 1D array of size M.
			:obj:`numpy.ndarray`
				Forces stored in an 2D arry of size M x 3N.
		"""

		if r.ndim == 2 and r.shape[0] > 1:
			return self._predict_bulk(r)

		r = r.reshape(self.n_atoms, 3)
		pdist = scipy.spatial.distance.pdist(r, 'euclidean')
		pdist = scipy.spatial.distance.squareform(pdist)

		r_desc = desc.r_to_desc(r,pdist)
		r_d_desc = desc.r_to_d_desc(r,pdist)

		res = sum(self.pool.map(partial(predict_worker_cached, r_desc=r_desc, r_d_desc=r_d_desc), self.wkr_starts_stops))
		return (res[-1]+self.c).reshape(-1), res[:-1].reshape(1,-1)

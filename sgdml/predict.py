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

import sys

import scipy.spatial.distance
import numpy as np

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

def _predict(r, n_train, std, c, chunk_size):

	r = r.reshape(-1, 3)
	pdist = scipy.spatial.distance.pdist(r, 'euclidean')
	pdist = scipy.spatial.distance.squareform(pdist, checks=False)

	r_desc = desc.r_to_desc(r, pdist)
	
	res = _predict_wkr((0,n_train), chunk_size, r_desc)
	res *= std

	F = desc.r_to_d_desc_op(r,pdist,res[1:]).reshape(1,-1)
	return (res[0]+c).reshape(-1), F

def _predict_wkr(wkr_start_stop, chunk_size, r_desc):
	"""
	Compute part of a prediction.

	The workload will be processed in `b_size` chunks.

	Parameters
	----------
		wkr_start_stop : tuple of int
			Indices of first and last (exclusive) sum element.
		r_desc : numpy.ndarray
			1D array containing the descriptor for the query geometry.

	Returns
	-------
		:obj:`numpy.ndarray`
			Partial prediction of all force components and energy
			(appended to array as last element).
	"""

	global glob,sig,n_perms

	wkr_start, wkr_stop = wkr_start_stop

	R_desc_perms = np.frombuffer(glob['R_desc_perms']).reshape(glob['R_desc_perms_shape'])
	R_d_desc_alpha_perms = np.frombuffer(glob['R_d_desc_alpha_perms']).reshape(glob['R_d_desc_alpha_perms_shape'])
	
	if 'alphas_E_lin' in glob:
		alphas_E_lin = np.frombuffer(glob['alphas_E_lin']).reshape(glob['alphas_E_lin_shape'])

	dim_d = r_desc.shape[0]
	dim_c = chunk_size*n_perms


	# pre-allocation

	diff_ab_perms = np.empty((dim_c, dim_d))
	a_x2 = np.empty((dim_c,))
	mat52_base = np.empty((dim_c,))

	mat52_base_fact = 5./(3*sig**3)
	diag_scale_fact = 5./sig
	sqrt5 = np.sqrt(5.)


	E_F = np.zeros((dim_d+1,))
	F = E_F[1:]

	wkr_start *= n_perms
	wkr_stop *= n_perms

	b_start = wkr_start
	for b_stop in range(wkr_start+dim_c,wkr_stop,dim_c) + [wkr_stop]:

		rj_desc_perms = R_desc_perms[b_start:b_stop,:]
		rj_d_desc_alpha_perms = R_d_desc_alpha_perms[b_start:b_stop,:]

		#diff_ab_perms = r_desc - rj_desc_perms
		np.subtract(np.broadcast_to(r_desc,rj_desc_perms.shape), rj_desc_perms, out=diff_ab_perms)
		norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

		#mat52_base = np.exp(-norm_ab_perms / sig) * mat52_base_fact
		np.exp(-norm_ab_perms / sig, out=mat52_base)
		mat52_base *= mat52_base_fact
		#a_x2 = np.einsum('ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms) # colum wise dot product
		np.einsum('ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms, out=a_x2) # colum wise dot product
		#a_x2 = np.einsum('ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms) * mat52_base # colum wise dot product

		#F += np.linalg.multi_dot([a_x2 * mat52_base, diff_ab_perms, r_d_desc]) * diag_scale_fact
		F += (a_x2 * mat52_base).dot(diff_ab_perms) * diag_scale_fact
		#F += a_x2.dot(diff_ab_perms) * diag_scale_fact
		mat52_base *= norm_ab_perms + sig

		#F -= np.linalg.multi_dot([mat52_base, rj_d_desc_alpha_perms, r_d_desc])
		F -= mat52_base.dot(rj_d_desc_alpha_perms)
		E_F[0] += a_x2.dot(mat52_base) # this one
		#E_F[0] += np.sum(a_x2)

		if 'alphas_E_lin' in glob:

			K_fe = diff_ab_perms * mat52_base[:,None]
			F += alphas_E_lin[b_start:b_stop].dot(K_fe)

			K_ee = (1 + (norm_ab_perms/sig) * (1 + norm_ab_perms/(3*sig)) ) * np.exp(-norm_ab_perms / sig)
			E_F[0] += K_ee.dot(alphas_E_lin[b_start:b_stop])

		b_start = b_stop

	return E_F


class GDMLPredict:

	def __init__(self, model, batch_size=None, num_workers=1, max_processes=None):

		global glob,sig,n_perms

		self.n_atoms = model['z'].shape[0]
		n_tril = self.n_atoms*(self.n_atoms-1) / 2
		
		self.n_train = model['R_desc'].shape[1]
		sig = model['sig']
		
		self.std = model['std'] if 'std' in model else 1.
		self.c = model['c']

		n_perms = model['perms'].shape[0]

		# Precompute permuted training descriptors and its first derivatives multiplied with the coefficients (only needed for cached variant).
		R_desc_perms = np.reshape(np.tile(model['R_desc'].T, n_perms)[:,model['tril_perms_lin']], (self.n_train*n_perms,-1), order='F')
		R_desc_perms = np.swapaxes(R_desc_perms.reshape(n_perms,self.n_train,-1),0,1).reshape((self.n_train*n_perms,-1))
		glob['R_desc_perms'], glob['R_desc_perms_shape'] = share_array(R_desc_perms)

		R_d_desc_alpha_perms = np.reshape(np.tile(model['R_d_desc_alpha'], n_perms)[:,model['tril_perms_lin']], (self.n_train*n_perms,-1), order='F')
		R_d_desc_alpha_perms = np.swapaxes(R_d_desc_alpha_perms.reshape(n_perms,self.n_train,-1),0,1).reshape((self.n_train*n_perms,-1))
		glob['R_d_desc_alpha_perms'], glob['R_d_desc_alpha_perms_shape'] = share_array(R_d_desc_alpha_perms)

		if 'alphas_E' in model:
			alphas_E_lin = np.tile(model['alphas_E'][:,None], (1,n_perms)).ravel()
			glob['alphas_E_lin'], glob['alphas_E_lin_shape'] = share_array(alphas_E_lin)


		# Parallel processing configuration

		self._bulk_mp = False # Bulk predictions with multiple processes?

		# How many parallel processes?
		self._max_processes = max_processes
		if self._max_processes is None:
			self._max_processes = mp.cpu_count()
		self.pool = None
		self.set_num_workers(num_workers)

		# Size of chunks in which each parallel task will be processed (unit: number of training samples)
		# This parameter should be as large as possible, but it depends on the size of available memory.
		self.set_batch_size(batch_size)

	def __del__(self):
		if self.pool is not None:
			self.pool.terminate()


	## Public ##

	def set_num_workers(self, num_workers=None): # TODO: complain if chunk or worker parameters do not fit training data (this causes issues with the caching)!!
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
				Number of processes (maximum value is set if `None`).
		"""

		if self.pool is not None:
			self.pool.terminate()
			self.pool.join()
			self.pool = None

		self._num_workers = 1
		if num_workers is None or num_workers > 1:
			self.pool = mp.Pool(processes=num_workers)
			self._num_workers = self.pool._processes

		# Data ranges for processes
		wkr_starts = range(0,self.n_train,int(np.ceil(float(self.n_train)/self._num_workers)))
		wkr_stops = wkr_starts[1:] + [self.n_train]

		self.wkr_starts_stops = zip(wkr_starts, wkr_stops)

	def set_batch_size(self, batch_size=None): # TODO: complain if chunk or worker parameters do not fit training data (this causes issues with the caching)!!
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
				Chunk size (maximum value is set if `None`).
		"""

		if batch_size is None:
			batch_size = self.n_train

		self._chunk_size = batch_size

	def set_opt_num_workers_and_batch_size_fast(self, n_bulk=1, n_reps=3):
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
			n_bulk : int
				Number of geometries that will be passed to the `predict`
				function in each call (performance will be optimized for
				that exact use case).
			n_reps : int
				Number of repetitions (bigger value: more accurate,
				but also slower).

		Returns
		-------
			int
				Force and energy prediciton speed in geometries
				per second.
		"""

		best_results = []
		last_i = None

		#for j in range(3):
		for j in [0]:

			best_gps = 0
			gps_min = 0.
			
			best_params = 1,1
			
			r_dummy = np.random.rand(n_bulk,self.n_atoms*3)

			def _dummy_predict():
				self.predict(r_dummy)

			#print  # remove me

			bulk_mp_rng = [True, False] if n_bulk > 1 else [False]
			for bulk_mp in bulk_mp_rng:
				self._bulk_mp = bulk_mp

				if bulk_mp == False:
					last_i = 0

				num_workers_rng = range(self._max_processes, 1, -1) if bulk_mp else range(1,self._max_processes+1)

				#num_workers_rng_sizes = [batch_size for batch_size in batch_size_rng if min_batch_size % batch_size == 0]


				#for num_workers in range(min_num_workers,self._max_processes+1):
				for num_workers in num_workers_rng:
					if not bulk_mp and self.n_train % num_workers != 0:
						continue
					#if bulk_mp and n_bulk % num_workers != 0:
					#	continue
					self.set_num_workers(num_workers)

					best_gps = 0
					gps_rng = (np.inf,0.)

					min_batch_size = min(self.n_train,n_bulk) if bulk_mp else int(np.ceil(self.n_train/num_workers))
					batch_size_rng = range(min_batch_size, 0, -1)

					#for i in range(0,min_batch_size):
					batch_size_rng_sizes = [batch_size for batch_size in batch_size_rng if min_batch_size % batch_size == 0]

					#print batch_size_rng_sizes

					i_done = 0
					i_dir = 1
					i = 0 if last_i == None else last_i
					#i = 0
					while i < len(batch_size_rng_sizes):

						#print i

						batch_size = batch_size_rng_sizes[i]
						self.set_batch_size(batch_size)

						i_done += 1

						gps = n_bulk * n_reps / (timeit.timeit(_dummy_predict, number=n_reps))
						#print '{:2d}@{:d} {:d} | {:7.2f} gps'.format(num_workers, batch_size, bulk_mp, gps)
						gps_rng = min(gps_rng[0],gps), max(gps_rng[1],gps)


						# gps still going up?
						# AND: gps not lower than the lowest overall?
						if gps < best_gps\
						and gps >= gps_min:
							if i_dir > 0 and i_done == 2 and batch_size != batch_size_rng_sizes[1]: # do we turn?
								i -= 2*i_dir
								i_dir = -1
								#print '><'
								continue
							else:
								#if batch_size == batch_size_rng_sizes[1]:
								#	i -= 1*i_dir
								#print '>>break ' + str(i_done)
								break
						else:
							best_gps = gps
							best_params = num_workers, batch_size, bulk_mp

						#if gps < best_gps:
						#	break
						#else:
						#	best_gps = gps
						#	best_params = num_workers, batch_size, bulk_mp

						if not bulk_mp and n_bulk > 1: # stop search early when multiple cpus are available and the 1 cpu case is tested
							if gps < gps_min: # if the batch size run is lower than the lowest overall, stop right here
								break

						i += 1*i_dir

					last_i = i - 1*i_dir
					i_dir = 1
					
					if len(best_results) > 0:
						overall_best_gps = max(best_results, key=lambda x:x[1])[1]
						if best_gps < overall_best_gps:
							break
						
						if best_gps < gps_min:
							break

					gps_min = gps_rng[0]
					#print 'gps_min ' + str(gps_min)

					#print 'best_gps'
					#print best_gps
					
					if len(best_results) > 0 and best_gps < overall_best_gps:
						break

					best_results.append((best_params, best_gps))

				
		(num_workers, batch_size, bulk_mp), gps = max(best_results, key=lambda x:x[1])

		self.set_batch_size(batch_size)
		self.set_num_workers(num_workers)
		self._bulk_mp = bulk_mp

		return gps

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

		if self._bulk_mp == True:
			for i,E_F in enumerate(self.pool.imap(partial(_predict, n_train=self.n_train, std=self.std, c=self.c, chunk_size=self._chunk_size), R)):
				E[i],F[i,:] = E_F
		else:
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
		pdist = scipy.spatial.distance.squareform(pdist, checks=False)

		r_desc = desc.r_to_desc(r, pdist)

		if self._num_workers == 1 or self._bulk_mp:
			res = _predict_wkr(self.wkr_starts_stops[0], self._chunk_size, r_desc)
		else:
			res = sum(self.pool.map(partial(_predict_wkr, chunk_size=self._chunk_size, r_desc=r_desc), self.wkr_starts_stops))
		res *= self.std

		E = res[0].reshape(-1) + self.c
		F = desc.r_to_d_desc_op(r,pdist,res[1:]).reshape(1,-1)
		return E, F

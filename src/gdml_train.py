"""
This module contains all routines for training sGDML and GDML models.
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

VERSION = 30918

import os, sys
import warnings

import scipy as sp
import numpy as np

from gdml_predict import GDMLPredict
from utils import desc,perm,io,ui

import multiprocessing as mp
from functools import partial

import timeit


glob = {}

def share_array(arr_np, typecode_or_type):
	"""
	Return a ctypes array allocated from shared memory with data from a
	NumPy array.

	Parameters
	----------
		arr_np : numpy.ndarray
			NumPy array.
		typecode_or_type : char or ctypes
			Either a ctypes type or a one character typecode of the
			kind used by the Python array module.

	Returns
	-------
		out : array of ctype
	"""

	arr = mp.RawArray(typecode_or_type, arr_np.ravel())
	return arr, arr_np.shape

def _assemble_kernel_mat_wkr(j, tril_perms_lin, n_perms, sig):
	"""
	Compute one row and column of the force field kernel matrix.

	The Hessian of the Matern kernel is used with n = 2 (twice
	differentiable). Each row and column consists of matrix-valued
	blocks, which encode the interaction of one training point with all
	others. The result is stored in shared memory (a global variable).

	Parameters
	----------
		j : int
			Index of training point.
		tril_perms_lin : numpy.ndarray
			1D array (int) containing all recovered permutations
			expanded as one large permutation to be applied to a tiled
			copy of the object to be permuted.
		n_perms : int
			Number of individual permutations encoded in 'tril_perms_lin'.
		sig : int
			Hyper-parameter sigma.

	Returns
	-------
		out : int
			Number of kernel matrix blocks created, divided by 2
			(symmetric blocks are always created at together).
	"""
		
	global glob

	R_desc = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
	R_d_desc = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])

	K = np.frombuffer(glob['K']).reshape(glob['K_shape'])

	n_train, dim_d, dim_i = R_d_desc.shape

	mat52_base_div = 3. * sig**4
	sqrt5 = np.sqrt(5.)
	sig_pow2 = sig**2

	base = np.arange(dim_i) # base set of indices
	blk_j = base + j * dim_i

	# Create permutated variants of 'rj_desc' and 'rj_d_desc'.
	rj_desc_perms = np.reshape(np.tile(R_desc[j,:], n_perms)[tril_perms_lin], (n_perms,-1), order='F')
	rj_d_desc_perms = np.reshape(np.tile(R_d_desc[j,:,:].T, n_perms)[:,tril_perms_lin], (-1,dim_d,n_perms))

	for i in range(j,n_train):

		blk_i = base[:, np.newaxis] + i * dim_i

		diff_ab_perms = R_desc[i,:] - rj_desc_perms
		norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

		mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5.
		diff_ab_outer_perms = 5. * np.einsum('ki,kj->ij', diff_ab_perms * mat52_base_perms[:,None], np.einsum('ik,jki -> ij', diff_ab_perms, rj_d_desc_perms))
		diff_ab_outer_perms -= np.einsum('ijk,k->ji', rj_d_desc_perms, ((sig_pow2 + sig * norm_ab_perms) * mat52_base_perms))

		K[blk_i,blk_j] = K[blk_j,blk_i] = R_d_desc[i,:,:].T.dot(diff_ab_outer_perms)

	return n_train - j


class GDMLTrain:

	def __init__(self, max_processes=None):
		self._max_processes = max_processes

	def create_task(self, train_dataset, n_train, test_dataset, n_test, sig, lam=1e-15, recov_sym=True):
		"""
		Create a data structure of custom type 'task'.

		These data structures serve as recipes for model creation, summarizing
		the configuration of one particular training run. Training and test
		points are sampled from the provided dataset, without replacement. If
		the same dataset if given for training and testing, the subsets are
		drawn without overlap.
		
		Each task also contains a choice for the hyper-parameters of the
		training process and the MD5 fingerprints of the used datasets.

		Parameters
		----------
			train_dataset : dict
				Data structure of custom type 'dataset' containing train dataset.
			n_train : int
				Number of training points to sample.
			test_dataset : dict
				Data structure of custom type 'dataset' containing test dataset.
			n_test : int
				Number of training points to sample.
			sig : int
				Hyper-parameter (kernel length scale).
			lam : float, optional
				Hyper-parameter lambda (regularization strength).
			recov_sym : bool, optional
				True: include symmetries (sGDML), False: GDML

		Returns
		-------
			out : dict
				Data structure of custom type 'task'.
		"""

		train_md5 = io.dataset_md5(train_dataset)
		test_md5 = io.dataset_md5(test_dataset)

		train_idxs = self.draw_strat_sample(train_dataset['E'], n_train)

		excl_idxs = train_idxs if train_md5 == test_md5 else None
		test_idxs = self.draw_strat_sample(test_dataset['E'], n_test, excl_idxs)

		R_train = train_dataset['R'][train_idxs,:,:]
		task = {'type':				't',\
				'dataset_name':		train_dataset['name'],\
				'dataset_theory':	train_dataset['theory'],\
				'z':				train_dataset['z'],\
				'R_train':			R_train,\
				'E_train':			train_dataset['E'][train_idxs],\
				'F_train':			train_dataset['F'][train_idxs,:,:],\
				'train_idxs':		train_idxs,\
				'train_md5':		train_md5,\
				'test_idxs':		test_idxs,\
				'test_md5':			test_md5,\
				'sig':				sig,\
				'lam':				lam}

		if recov_sym:
			task['perms'] 		= perm.sync_mat(R_train, train_dataset['z'], self._max_processes)
			task['perms'] 		= perm.complete_group(task['perms'])
		else:
			task['perms'] 		= np.arange(train_dataset['R'].shape[1])[None,:] # no symmetries

		return task

	def train(self, task):
		"""
		Train a model based on a training task.

		Parameters
		----------
			task : dict
				Data structure of custom type 'task'.

		Returns
		-------
			out : dict
				Data structure of custom type 'model'.
		"""

		sig = np.squeeze(task['sig'])
		lam = np.squeeze(task['lam'])

		n_perms = task['perms'].shape[0]
		tril_perms = np.array([perm.to_tril_perm(p) for p in task['perms']]);
		perm_offsets = np.arange(n_perms)[:,None] * tril_perms.shape[1]
		tril_perms_lin = (tril_perms + perm_offsets).flatten('F')

		n_train, n_atoms = task['R_train'].shape[:2]
		dim_i = n_atoms * 3
		dim_d = (n_atoms**2 - n_atoms) / 2

		R_desc = np.empty([n_train, dim_d])
		R_d_desc = np.empty([n_train, dim_d, dim_i])

		for i in range(n_train):
			r = task['R_train'][i]

			pdist = sp.spatial.distance.pdist(r,'euclidean')
			pdist = sp.spatial.distance.squareform(pdist)

			R_desc[i,:] = desc.r_to_desc(r,pdist)
			R_d_desc[i,:,:] = desc.r_to_d_desc(r,pdist)

		Ft = task['F_train'].ravel()

		start = timeit.default_timer()
		K = self._assemble_kernel_mat(R_desc, R_d_desc, tril_perms_lin, n_perms, sig)
		stop = timeit.default_timer()
		print " \x1b[90m(%.1f s)\x1b[0m" % ((stop - start) / 2)

		sys.stdout.write('\r[' + ui.blink_str(' .. ') + '] Solving linear system...')
	 	sys.stdout.flush()

		start = timeit.default_timer()

		K[np.diag_indices_from(K)] -= lam # regularizer
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			alphas = sp.linalg.solve(K, Ft, overwrite_a=True, overwrite_b=True, check_finite=False)

		stop = timeit.default_timer()
		sys.stdout.write('\r[DONE] Solving linear system...    \x1b[90m(%.1f s)\x1b[0m\n' % ((stop - start) / 2))
	 	sys.stdout.flush()

		r_dim = R_d_desc.shape[2]
		r_d_desc_alpha = [rj_d_desc.dot(alphas[(j * r_dim):((j + 1) * r_dim)]) for j,rj_d_desc in enumerate(R_d_desc)]

		model = {'type': 			'm',\
				 'dataset_name':	np.squeeze(task['dataset_name']),\
				 'dataset_theory':	np.squeeze(task['dataset_theory']),\
				 'z':				task['z'],\
				 'train_idxs':		task['train_idxs'],\
				 'train_md5':		task['train_md5'],\
				 'test_idxs':		task['test_idxs'],\
				 'test_md5':		task['test_md5'],\
				 'n_valid':			0,\
				 'valid_md5':		None,\
				 'e_err':			{'mae':np.nan, 'rmse':np.nan},\
				 'f_err':			{'mae':np.nan, 'rmse':np.nan},\
				 'R_desc':			R_desc.T,\
				 'R_d_desc_alpha': 	r_d_desc_alpha,\
				 'c':				0.,\
				 'sig': 			sig,\
				 'perms': 			task['perms'],\
				 'tril_perms_lin':	tril_perms_lin}

	 	model['c'] = self._recov_int_const(model, task)

		return model

	def _recov_int_const(self, model, task):
		"""
		Estimate the integration constant for a force field model.

		The offset between the energies predicted for the original training
		data and the true energy labels is computed in the least square sense.
		Furthermore, common issues with the user-provided datasets are self
		diagnosed here.

		Parameters
		----------
			model : dict
				Data structure of custom type 'model'.
			task : dict
				Data structure of custom type 'task'.

		Returns
		-------
			out : float
				Estimate for the integration constant.

		Raises
		------
			ValueError
				If the sign of the force labels in the dataset from
				which the model emerged is switched (e.g. gradients
				instead of forces).
		"""

		gdml = GDMLPredict(model)
		n_train = task['E_train'].shape[0]

		R = task['R_train'].reshape(n_train,-1)
		E_pred,_ = gdml.predict(R)
		E_ref = np.squeeze(task['E_train'])

		e_fact = np.linalg.lstsq(np.column_stack((E_pred, np.ones(E_ref.shape))), E_ref, rcond=-1)[0][0]

		if np.abs(e_fact - 1) > 1e-1:
			print ui.warn_str('[WARN]') + ' Provided dataset uses inconsistent energy units! Integrated forces differ from energy labels by factor ~%.2E.' % e_fact +\
							  '\n       This can have several reasons: wrong unit conversion, inaccurate force labels, etc.'
			#raise ValueError('Provided dataset uses inconsistent energy units! Integrated forces differ from energy labels by factor ~%.2E.' % e_fact\
			#			   + '\n       A variation of this factor over different training sets indicates a problem with the force labels instead.')

		if np.sign(e_fact) == -1:
			raise ValueError('Provided dataset contains gradients instead of force labels (flipped sign).')

		#c22 = np.sum(E_ref - E_pred) / E_ref.shape[0]
		#import matplotlib.pyplot as plt
	 	#plt.plot(range(len(E_pred)), E_pred+c22, 'b')
	 	#plt.plot(range(len(E_pred)), E_ref, 'g')
	 	#plt.show()

		# Least squares estimate for integration constant.
		return np.sum(E_ref - E_pred) / E_ref.shape[0]

	def _assemble_kernel_mat(self, R_desc, R_d_desc, tril_perms_lin, n_perms, sig):
		"""
		Compute force field kernel matrix.

		The Hessian of the Matern kernel is used with n = 2 (twice
		differentiable). Each row and column consists of matrix-valued blocks,
		which encode the interaction of one training point with all others. The
		result is stored in shared memory (a global variable).

		Parameters
		----------
			R_desc : numpy.ndarray
				Array containing the descriptor for each training point.
			R_d_desc : numpy.ndarray
				Array containing the gradient of the descriptor for
				each training point.
			tril_perms_lin : numpy.ndarray
				1D array containing all recovered permutations
				expanded as one large permutation to be applied to a
				tiled copy of the object to be permuted.
			n_perms : int
				Number of individual permutations encoded in
				'tril_perms_lin'.
			sig : int
				Hyper-parameter (kernel length scale).

		Returns
		-------
			out : numpy.ndarray
				Force field kernel matrix.
		"""

		global glob

		n_train, dim_d, dim_i = R_d_desc.shape

		K = mp.RawArray('d', (n_train*dim_i)**2)
		glob['K'], glob['K_shape'] = K, (n_train*dim_i, n_train*dim_i)

		glob['R_desc'], glob['R_desc_shape'] = share_array(R_desc, 'd')
		glob['R_d_desc'], glob['R_d_desc_shape'] = share_array(R_d_desc, 'd')

		pool = mp.Pool(self._max_processes)
		done_total = 0
		for done in pool.imap_unordered(partial(_assemble_kernel_mat_wkr, tril_perms_lin=tril_perms_lin, n_perms=n_perms, sig=sig), range(n_train)):			
			done_total += done

	 		progr = float(done_total) / ((n_train**2 - n_train) / 2 + n_train)
	 		sys.stdout.write('\r[%3d%%] Assembling kernel matrix...' % (progr * 100))
	 		sys.stdout.flush()
	 	pool.close()

	 	# Release some memory.
	 	glob.pop('K', None)
	 	glob.pop('R_desc', None)
	 	glob.pop('R_d_desc', None)

		return np.frombuffer(K).reshape(glob['K_shape'])

	def draw_strat_sample(self, T, n, excl_idxs=None):
		"""
		Draw sample from dataset that preserves its original distribution.

		The distribution is estimated from a histogram were the bin size is
		determined using the Freedman-Diaconis rule. This rule is designed to
		minimize the difference between the area under the empirical
		probability distribution and the area under the theoretical
		probability distribution. A reduced histogram is then constructed by
		sampling uniformly in each bin. It is intended to populate all bins
		with at least one sample in the reduced histogram, even for small
		training sizes.

		Parameters
		----------
			T : numpy.ndarray
				Dataset to sample from.
			n : int
				Number of samples.
			excl_idxs : numpy.ndarray, optional
				Array of indices to exclude from sample.

		Returns
		-------
			out : numpy.ndarray
				Array of indices that form the sample.
		"""

		n_train = T.shape[0]

		# Freedman-Diaconis rule
		h = 2*np.subtract(*np.percentile(T, [75, 25])) / np.cbrt(n)
		n_bins = int(np.ceil((np.max(T)-np.min(T)) / h))

		bins = np.linspace(np.min(T), np.max(T), n_bins, endpoint=False)
		idxs = np.digitize(T, bins)

		# Exclude restricted indices.
		if excl_idxs is not None:
			idxs[excl_idxs] = n_bins+1 # Impossible bin.

		uniq_all,cnts_all = np.unique(idxs, return_counts=True)

		# Remove restricted bin.
		if excl_idxs is not None:
			excl_bin_idx = np.where(uniq_all == n_bins+1)
			cnts_all = np.delete(cnts_all, excl_bin_idx)
			uniq_all = np.delete(uniq_all, excl_bin_idx)

		# Compute reduced bin counts.
		reduced_cnts = np.ceil(cnts_all/np.sum(cnts_all, dtype=float) * n).astype(int)
		reduced_cnts = np.minimum(reduced_cnts, cnts_all) # limit reduced_cnts to what is available in cnts_all

		# Reduce/increase bin counts to desired total number of points.
		reduced_cnts_delta = n - np.sum(reduced_cnts)

		# Draw additional bin members to fill up/drain bucket counts of subset. This array contains (repeated) bucket IDs.
		outstanding = np.random.choice(uniq_all, np.abs(reduced_cnts_delta), p=(reduced_cnts-1)/np.sum(reduced_cnts-1, dtype=float))
		uniq_outstanding,cnts_outstanding = np.unique(outstanding, return_counts=True) # Aggregate bucket IDs.

		outstanding_bucket_idx = np.where(np.in1d(uniq_all, uniq_outstanding))[0] # Bucket IDs to Idxs.
		reduced_cnts[outstanding_bucket_idx] += np.sign(reduced_cnts_delta)*cnts_outstanding

		train_idxs = np.empty((0,), dtype=int)
		for uniq_idx, bin_cnt in zip(uniq_all, reduced_cnts):
			idx_in_bin_all = np.where(idxs.ravel() == uniq_idx)[0]
			train_idxs = np.append(train_idxs, np.random.choice(idx_in_bin_all, bin_cnt, replace=False))
		return train_idxs
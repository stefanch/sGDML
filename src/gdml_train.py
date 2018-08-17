# GDML Force Field
# Author: Stefan Chmiela (stefan@chmiela.com)
VERSION = 180718

import os, sys

import scipy as sp
import numpy as np

from gdml_predict import GDMLPredict
from utils import desc,perm,io,ui

import multiprocessing as mp
from functools import partial

import timeit


glob = {}

def share_array(arr_np, typecode):
	arr = mp.RawArray(typecode, arr_np.ravel())
	return arr, arr_np.shape

def _assemble_kernel_mat_wkr(j, tril_perms_lin, n_perms, sig, lam):
		
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

		#diff_ab_rj_d_desc_perms = np.einsum('ik,jki -> ij', diff_ab_perms, rj_d_desc_perms)
		mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5.
		#diff_ab_outer_perms = 5 * np.einsum('ki,kj->ij', diff_ab_perms * mat52_base_perms[:,None], diff_ab_rj_d_desc_perms)
		diff_ab_outer_perms = 5. * np.einsum('ki,kj->ij', diff_ab_perms * mat52_base_perms[:,None], np.einsum('ik,jki -> ij', diff_ab_perms, rj_d_desc_perms))
		diff_ab_outer_perms -= np.einsum('ijk,k->ji', rj_d_desc_perms, ((sig_pow2 + sig * norm_ab_perms) * mat52_base_perms))

		K[blk_i,blk_j] = K[blk_j,blk_i] = R_d_desc[i,:,:].T.dot(diff_ab_outer_perms)

	return n_train - j


class GDMLTrain:

	def create_task(self, train_dataset, n_train, test_dataset, n_test, sig, lam=1e-15, recov_sym=True):

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
			task['perms'] 		= perm.sync_mat(R_train, train_dataset['z'])
			task['perms'] 		= perm.complete_group(task['perms'])
		else:
			task['perms'] 		= np.arange(train_dataset['R'].shape[1])[None,:] # no symmetries

		return task

	def train(self, task):

		sig = np.squeeze(task['sig'])
		lam = np.squeeze(task['lam'])

		n_perms = task['perms'].shape[0]
		tril_perms = np.array([perm.to_tril_perm(p) for p in task['perms']]);
		perm_offsets = np.arange(n_perms)[:,None] * tril_perms.shape[1]
		tril_perms_lin = (tril_perms + perm_offsets).flatten('F')

		n_train, n_atoms = task['R_train'].shape[:2]
		dim_i = n_atoms * 3
		dim_d = (n_atoms**2 - n_atoms) / 2

		R_desc = np.empty([n_train, dim_d]) # TODO
		R_d_desc = np.empty([n_train, dim_d, dim_i]) # TODO

		for i in range(n_train):
			r = task['R_train'][i]

			pdist = sp.spatial.distance.pdist(r,'euclidean')
			pdist = sp.spatial.distance.squareform(pdist)

			R_desc[i,:] = desc.r_to_desc(r,pdist)
			R_d_desc[i,:,:] = desc.r_to_d_desc(r,pdist)

		Ft = task['F_train'].ravel()

		start = timeit.default_timer()
		K = self._assemble_kernel_mat(R_desc, R_d_desc, tril_perms_lin, n_perms, sig, lam)
		stop = timeit.default_timer()
		print " \x1b[90m(%.1f s)\x1b[0m" % ((stop - start) / 2)

		sys.stdout.write('\r[' + ui.blink_str(' .. ') + '] Solving linear system...')
	 	sys.stdout.flush()

		start = timeit.default_timer()

		K[np.diag_indices_from(K)] -= lam # regularizer
		#alphas = np.linalg.solve(K, Ft)
		
		alphas = sp.linalg.solve(K, Ft, overwrite_a=True, overwrite_b=True, check_finite=False)

		stop = timeit.default_timer()
		sys.stdout.write('\r[DONE] Solving linear system...    \x1b[90m(%.1f s)\x1b[0m\n' % ((stop - start) / 2))
	 	sys.stdout.flush()

		# Do some preprocessing.
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

		return model

	def recov_int_const(self, model, task, tol=1e-7):

		gdml = GDMLPredict(model)
		n_train = task['E_train'].shape[0]

		R = task['R_train'].reshape(n_train,-1)
		E_pred,_ = gdml.predict(R)
		E_ref = np.squeeze(task['E_train'])
		diff = E_ref - E_pred
		
		c_range = np.linspace(min(diff), max(diff), num=100)

		curr_mae = np.inf
		i = 0
		while True:
			c = c_range[i]

			last_mae = curr_mae
			curr_mae = np.sum(abs((E_pred + c) - E_ref)) / diff.shape[0]
			if curr_mae > last_mae and i > 0:
				if (curr_mae - last_mae) <= tol:
					return c

				c_range = np.linspace(c_range[i-1], c, num=100)
				curr_mae = np.inf
				i = 0
			i += 1

	def _assemble_kernel_mat(self, R_desc, R_d_desc, tril_perms_lin, n_perms, sig, lam):

		global glob

		n_train, dim_d, dim_i = R_d_desc.shape

		K = mp.RawArray('d', (n_train*dim_i)**2)
		glob['K'], glob['K_shape'] = K, (n_train*dim_i, n_train*dim_i)

		glob['R_desc'], glob['R_desc_shape'] = share_array(R_desc, 'd')
		glob['R_d_desc'], glob['R_d_desc_shape'] = share_array(R_d_desc, 'd')

		pool = mp.Pool()
		done_total = 0
		for done in pool.imap_unordered(partial(_assemble_kernel_mat_wkr, tril_perms_lin=tril_perms_lin, n_perms=n_perms, sig=sig, lam=lam), range(n_train)):			
			done_total += done

	 		progr = float(done_total) / ((n_train**2 - n_train) / 2 + n_train)
	 		sys.stdout.write('\r[%3d%%] Assembling kernel matrix...' % (progr * 100))
	 		sys.stdout.flush()
	 	pool.close()

		return np.frombuffer(K).reshape(glob['K_shape'])

	def draw_strat_sample(self, T, n, excl_idxs=None):

		n_train = T.shape[0]

		# Freedman-Diaconis rule
		h = 2*np.subtract(*np.percentile(T, [75, 25])) / np.cbrt(n)
		n_bins = int(np.ceil((np.max(T)-np.min(T)) / h))

		bins = np.linspace(np.min(T), np.max(T), n_bins, endpoint=False)
		idxs = np.digitize(T, bins)

		# Exlude restricted indices.
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
#!/usr/bin/python

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

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import argparse
import time
import shutil
from functools import partial

import numpy as np

from sgdml import __version__
from train import GDMLTrain
from predict import GDMLPredict

from utils import io,ui

PACKAGE_NAME = 'sgdml'


class AssistantError(Exception):
    pass

def _print_splash():
	print """         __________  __  _____ 
   _____/ ____/ __ \/  |/  / / 
  / ___/ / __/ / / / /|_/ / /  
 (__  ) /_/ / /_/ / /  / / /___
/____/\____/_____/_/  /_/_____/  """ + __version__

def _print_dataset_properties(dataset):

	n_mols, n_atoms, _ = dataset['R'].shape
	print ' {:<16} {:<} ({:<d} atoms)'.format('Name:', dataset['name'], n_atoms)
	print ' {:<16} {:<}'.format('Theory:', dataset['theory'])
	print ' {:<16} {:<d}'.format('Size:', n_mols)

	if 'E' in dataset:
		T_min, T_max = np.min(dataset['E']), np.max(dataset['E'])
		print ' {:<16} {:<.3} '.format('Energies:', T_min) + '|--' + ' {:^8.3} '.format(T_max-T_min) + '--|' + ' {:>9.3} [a.u.]'.format(T_max)
	else:
		print ' {:<16} {:<}'.format('Energies:', 'n/a')

	TG_min, TG_max = np.min(dataset['F'].ravel()), np.max(dataset['F'].ravel())
	print ' {:<16} {:<.3} '.format('Forces:', TG_min) + '|--' + ' {:.3} '.format(TG_max-TG_min) + '--|' + ' {:>9.3} [a.u.]'.format(TG_max)

	print ' {:<16} {:<}'.format('Fingerprint:', dataset['md5'])
	print

def _print_task_properties(use_sym, use_cprsn, use_E, use_E_cstr):

	energy_fix_str = ('kernel constraints (+E)' if use_E_cstr else 'global integration constant recovery') if use_E else 'none'
	print ' {:<16} {:<}'.format('Energy handling:', energy_fix_str)

	print ' {:<16} {:<}'.format('Symmetries:', 'include (sGDML)' if use_sym else 'ignore (GDML)')
	print ' {:<16} {:<}'.format('Compression:', 'requested' if use_cprsn else 'not requested')

	print

def _print_model_properties(model):
	
	print ' {:<16} {:<d}'.format('Symmetries:', len(model['perms']))

	n_train = len(model['idxs_train'])
	print ' {:<16} {:<d} points from \'{:<}\''.format('Trained on:', n_train, model['md5_train'])

	if model['use_E']:
		e_err = model['e_err'].item()
	f_err = model['f_err'].item()

	n_valid = len(model['idxs_valid'])
	#is_tested = not np.isnan(e_err['mae']) and not np.isnan(e_err['rmse']) and not np.isnan(f_err['mae']) and not np.isnan(f_err['rmse'])
	is_valid = not np.isnan(f_err['mae']) and not np.isnan(f_err['rmse'])
	print ' {:<16} {}{:<d} points from \'{:<}\''.format('Validated on:', '' if is_valid else '[pending] ', n_valid, model['md5_valid'])

	n_test = int(model['n_test'])
	is_test = n_test > 0
	if is_test:
		print ' {:<16} {:<d} points from \'{:<}\''.format('Tested on:', n_test, model['md5_test'])
	else:
		print ' {:<16} {}'.format('Test:', '[pending]')
	#print ' {:<14} {}'.format('Test:', '{} points'.format(n_valid) if is_valid else '[pending]')

	if is_valid:
		action_str = 'Validation' if not is_valid else 'Expected test'
		print ' {:<16}'.format('{} errors:'.format(action_str))
		if model['use_E']:
			print '  {:<15} {:>.2e}/{:>.2e} [a.u.]'.format('Energy:', e_err['mae'], e_err['rmse'])
		print '  {:<15} {:>.2e}/{:>.2e} [a.u.]'.format('Forces:', f_err['mae'], f_err['rmse'])
	print


def all(dataset, valid_dataset, test_dataset, n_train, n_valid, n_test, sigs, gdml, use_E, use_E_cstr, use_cprsn, overwrite, max_processes, **kwargs):

	print '\n' + ui.white_back_str(' STEP 0 ') + ' Dataset(s)\n' + '-'*100

	print ui.white_bold_str('Properties')
	_, dataset_extracted = dataset
	_print_dataset_properties(dataset_extracted)

	if valid_dataset is not None:
		print ui.white_bold_str('Properties (validation)')
		_, valid_dataset_extracted = valid_dataset
		_print_dataset_properties(valid_dataset_extracted)

		if not np.array_equal(dataset_extracted['z'], valid_dataset_extracted['z']):
			raise AssistantError('Atom composition or order in validation dataset does not match the one in bulk dataset.')

	if test_dataset is not None:
		print ui.white_bold_str('Properties (test)')
		_, test_dataset_extracted = test_dataset
		_print_dataset_properties(test_dataset_extracted)

		if not np.array_equal(dataset_extracted['z'], test_dataset_extracted['z']):
			raise AssistantError('Atom composition or order in test dataset does not match the one in bulk dataset.')

	print ui.white_back_str(' STEP 1 ') + ' Cross-validation task creation\n' + '-'*100
	task_dir = create(dataset, test_dataset, n_train, n_valid, sigs, gdml, use_E, use_E_cstr, use_cprsn, overwrite, max_processes, **kwargs)

	print ui.white_back_str(' STEP 2 ') + ' Training\n' + '-'*100
	task_dir_arg = ui.is_dir_with_file_type(task_dir, 'task')
	model_dir_or_file_path = train(task_dir_arg, overwrite, max_processes, **kwargs)

	print ui.white_back_str(' STEP 3 ') + ' Validation\n' + '-'*100
	model_dir_arg = ui.is_dir_with_file_type(model_dir_or_file_path, 'model', or_file=True)
	if valid_dataset is None: valid_dataset = dataset
	validate(model_dir_arg, valid_dataset, overwrite=False,max_processes=max_processes, **kwargs)

	print ui.white_back_str(' STEP 4 ') + ' Hyper-parameter selection\n' + '-'*100
	#if sigs is None or len(sigs) > 1: # Skip testing and selection, if only one model was trained.
	model_file_name = select(model_dir_arg, overwrite, max_processes, **kwargs)
	#else:
	#	best_model_path = model_dir_or_file_path # model_dir_or_file_path = model_path, if single model is being trained
	#	print ui.info_str('[INFO]') + ' Skipping step because only one model is being trained.\n'

	print ui.white_back_str(' STEP 5 ') + ' Test.\n' + '-'*100
	model_dir_arg = ui.is_dir_with_file_type(model_file_name, 'model', or_file=True)
	if test_dataset is None: test_dataset = dataset
	test(model_dir_arg, test_dataset, n_test, overwrite=False, max_processes=max_processes, **kwargs)

	print ui.pass_str('[DONE]') + ' Training assistant finished sucessfully.'
	print '       Here is your model: \'%s\'' % model_file_name


# if training job exists and is a subset of the requested cv range, add new tasks
# otherwise, if new range is different or smaller, fail
def create(dataset, valid_dataset, n_train, n_valid, sigs, gdml, use_E, use_E_cstr, use_cprsn, overwrite, max_processes, command=None, **kwargs):

	dataset_path, dataset = dataset
	n_data = dataset['F'].shape[0]

	func_called_directly = command == 'create' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n TASK CREATION \n') + '-'*100
		print ui.white_bold_str('Dataset properties')
		_print_dataset_properties(dataset)

	print ui.white_bold_str('Properties')
	_print_task_properties(use_sym=not gdml, use_cprsn=use_cprsn, use_E=use_E, use_E_cstr=use_E_cstr)

	if n_data < n_train:
		raise AssistantError('Dataset only contains {} points, can not train on {}.'.format(n_data,n_train))

	if valid_dataset is None:
		valid_dataset_path, valid_dataset = dataset_path, dataset
		if n_data - n_train < n_valid:
			raise AssistantError('Dataset only contains {} points, can not train on {} and validate on {}.'.format(n_data,n_train,n_valid))
	else:
		valid_dataset_path, valid_dataset = valid_dataset
		n_valid_data = valid_dataset['E'].shape[0]
		if n_valid_data < n_valid:
			raise AssistantError('Validation dataset only contains {} points, can not validate on {}.'.format(n_data,n_valid))

	lam = 1e-15
	if sigs is None:
		print ui.info_str('[INFO]') + ' Kernel hyper-paramter sigma was automatically set to range \'2:10:100\'.'
		sigs = range(2,100,10) # default range

	gdml_train = GDMLTrain(max_processes=max_processes)

	task_dir = io.train_dir_name(dataset, n_train, use_sym=not gdml, use_cprsn=use_cprsn, use_E=use_E, use_E_cstr=use_E_cstr)
	task_file_names = []
	if os.path.exists(task_dir):
		if overwrite:
			print ui.info_str('[INFO]') + ' Overwriting existing training directory.'
			shutil.rmtree(task_dir)
			os.makedirs(task_dir)
		else:
			if ui.is_task_dir_resumeable(task_dir, dataset, valid_dataset, n_train, n_valid, sigs, gdml):
				print ui.info_str('[INFO]') + ' Resuming existing hyper-parameter search in \'%s\'.' % task_dir
				
				# Get all task file names.
				try:
					_, task_file_names = ui.is_dir_with_file_type(task_dir, 'task')
				except:
   					pass
			else:
				raise AssistantError('Unfinished hyper-parameter search found in \'%s\'.' % task_dir\
					+ '\n       Run \'%s %s -o %s %d %d -s %s\' to overwrite.' % (PACKAGE_NAME, command, dataset_path, n_train, n_valid, ' '.join(str(s) for s in sigs)))
	else:
		os.makedirs(task_dir)
	
	if task_file_names:

		with np.load(os.path.join(task_dir, task_file_names[0])) as task:
			tmpl_task = dict(task)
		#tmpl_task = dict(np.load(os.path.join(task_dir, task_file_names[0])))
	else:
		if not use_E:
			print ui.info_str('[INFO]') + ' Energy labels will be ignored for training.'
			print '       Note: If available in the dataset file, the energy labels will however still be used to '
			print '             generate stratified training, test and validation datasets. Otherwise a uniform '
			print '             sampling is used.'

		if 'E' not in dataset:
			print ui.warn_str('[WARN]') + ' Training dataset will be sampled with no guidance from energy labels (uniformly)!'

		if 'E' not in valid_dataset:
			print ui.warn_str('[WARN]') + ' Validation dataset will be sampled with no guidance from energy labels (uniformly)!'
			print '       Note: Larger validation datasets are recommended due to slower convergence of the error.'

		try:
			tmpl_task = gdml_train.create_task(dataset, n_train, valid_dataset, n_valid, sig=1, lam=lam, use_sym=not gdml, use_E=use_E, use_E_cstr=use_E_cstr, use_cprsn=use_cprsn) # template task
		except Exception, err:
			sys.exit(ui.fail_str('[FAIL]') + ' %s' % err)
		
	n_written = 0
	for sig in sigs:
		tmpl_task['sig'] = sig
		task_file_name = io.task_file_name(tmpl_task)
		task_path = os.path.join(task_dir, task_file_name)

		if os.path.isfile(task_path):
			print ui.warn_str('[WARN]') + ' Skipping existing task \'%s\'.' % task_file_name
		else:
			np.savez_compressed(task_path, **tmpl_task)
			n_written += 1
	if n_written > 0:
		print '[DONE] Writing %d/%d tasks with %s training points each.' % (n_written, len(sigs), tmpl_task['R_train'].shape[0])
	print ''

	if func_called_directly:
		print ui.white_back_str(' NEXT STEP ') + ' %s train %s\n' % (PACKAGE_NAME, task_dir)

	return task_dir


def train(task_dir, overwrite, max_processes, command=None, **kwargs):

	task_dir, task_file_names = task_dir
	n_tasks = len(task_file_names)

	func_called_directly = command == 'train' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n MODEL TRAINING \n') + '-'*100

	def cprsn_callback(n_atoms, n_atoms_kept):
		print ui.info_str('[INFO]') + ' %d out of %d atoms remain after compression.' % (n_atoms_kept, n_atoms)
		print '       Note: Compression reduces optimization problem size at the cost of prediction accuracy!'
	ker_progr_callback = partial(ui.progr_bar, disp_str='Assembling kernel matrix...')
	solve_callback = partial(ui.progr_toggle, disp_str='Solving linear system...   ')

	n_failed = 0

	gdml_train = GDMLTrain(max_processes=max_processes)
	for i,task_file_name in enumerate(task_file_names):
		if n_tasks > 1:
			print ui.white_bold_str('Training task %d of %d' % (i+1, n_tasks))

		task_file_path = os.path.join(task_dir, task_file_name)
		#task_file_relpath = os.path.relpath(task_file_path, BASE_DIR)
		with np.load(task_file_path) as task:

			model_file_name = io.model_file_name(task, is_extended=False)
			model_file_path = os.path.join(task_dir, model_file_name)
			#model_file_relpath = os.path.relpath(model_file_path, BASE_DIR)

			if not overwrite and os.path.isfile(model_file_path):
				print ui.warn_str('[WARN]') + ' Skipping exising model \'%s\'.' % model_file_name
				if func_called_directly:
					print '       Run \'%s train -o %s\' to overwrite.' % (PACKAGE_NAME, task_file_path)
				print
				continue

			try:
				model = gdml_train.train(task, cprsn_callback, ker_progr_callback, solve_callback)
			except Exception, err:
				sys.exit(ui.fail_str('[FAIL]') + ' %s' % err)
			else:
				if func_called_directly:
					print '[DONE] Writing model to file \'%s\'...' % model_file_path
				np.savez_compressed(model_file_path, **model)
			print

	model_dir_or_file_path = model_file_path if n_tasks == 1 else task_dir
	if func_called_directly:
		print ui.white_back_str(' NEXT STEP ') + ' %s validate %s %s\n' % (PACKAGE_NAME, model_dir_or_file_path, '<dataset_file>')

	return model_dir_or_file_path #model directory or file

def _batch(iterable, n=1):	
	l = len(iterable)
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]

def _online_err(err, size, n, mae_n_sum, rmse_n_sum):

	err = np.abs(err)

	mae_n_sum += np.sum(err) / size
	mae = mae_n_sum / n

	rmse_n_sum += np.sum(err**2) / size
	rmse = np.sqrt(rmse_n_sum / n)

	return mae, mae_n_sum, rmse, rmse_n_sum


def validate(model_dir, dataset, overwrite, max_processes, command=None, **kwargs):

	dataset_path_extracted, dataset_extracted = dataset

	func_called_directly = command == 'validate' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n MODEL VALIDATION \n') + '-'*100
		print ui.white_bold_str('Dataset properties')
		_print_dataset_properties(dataset_extracted)

	n_test = 0 # TODO remove?
	test(model_dir, dataset, n_test, overwrite, max_processes, command, **kwargs)

	if func_called_directly:
		model_dir, model_file_names = model_dir
		n_models = len(model_file_names)

		if n_models == 1:
			model_file_path = os.path.join(model_dir, model_file_names[0])
			print ui.white_back_str(' NEXT STEP ') + ' %s test %s %s %s\n' % (PACKAGE_NAME, model_file_path, dataset_path_extracted, '<n_test>')
		else:
			print ui.white_back_str(' NEXT STEP ') + ' %s select %s\n' % (PACKAGE_NAME, model_dir)


def test(model_dir, dataset, n_test, overwrite, max_processes, command=None, **kwargs):

	model_dir, model_file_names = model_dir
	n_models = len(model_file_names)

	dataset_path, dataset = dataset

	func_called_directly = command == 'test' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n MODEL TEST \n') + '-'*100
		print ui.white_bold_str('Dataset properties')
		_print_dataset_properties(dataset)

	num_workers, batch_size = 0,0
	for i,model_file_name in enumerate(model_file_names):

		model_path = os.path.join(model_dir, model_file_name)
		_, model = ui.is_file_type(model_path, 'model')

		#use_E = model['use_E'] if 'use_E' in model else True

		if i == 0 and command != 'all':
			print ui.white_bold_str('Model properties')
			_print_model_properties(model)

		if not np.array_equal(model['z'], dataset['z']):
			raise AssistantError('Atom composition or order in dataset does not match the one in model.')

		if model['use_E']:
			e_err = model['e_err'].item()
		f_err = model['f_err'].item()
	
		# is this a test or validation run?
		#needs_test = np.isnan(e_err['mae']) and np.isnan(e_err['rmse']) and np.isnan(f_err['mae']) and np.isnan(f_err['rmse'])
		needs_valid = np.isnan(f_err['mae']) and np.isnan(f_err['rmse'])
		#if model['use_E']:

		is_test = n_test != 0 and not needs_valid

		#print ui.white_bold_str(('%s model' % ('Validating' if is_valid else 'Testing')) + ('' if n_models == 1 else ' %d of %d' % (i+1, n_models)))
		if n_models > 1:
			print ui.white_bold_str('%s model %d of %d' % ('Testing' if is_test else 'Validating', i+1, n_models))

		if not overwrite and not needs_valid and not is_test:
			print ui.warn_str('[WARN]') + ' Skipping already validated model \'%s\'.' % model_file_name
			if command == 'test':
				print '       Run \'%s validate -o %s %s\' to overwrite.' % (PACKAGE_NAME, model_path, dataset_path)
			print
			continue

		# TODO:
		# (1) check if user tried to validate an untested model

		if needs_valid and dataset['md5'] != model['md5_valid']:
			raise AssistantError('Fingerprint of provided test dataset does not match the one in model file.')

		test_idxs = model['idxs_valid']
		if is_test:
			gdml = GDMLTrain(max_processes=max_processes)

			# exclude training and/or test sets from validation set if necessary
			excl_idxs = np.empty((0,), dtype=int)
			if dataset['md5'] == model['md5_train']:
				excl_idxs = np.concatenate([excl_idxs, model['idxs_train']])
			if dataset['md5'] == model['md5_valid']:
				excl_idxs = np.concatenate([excl_idxs, model['idxs_valid']])
			if len(excl_idxs) == 0:
				excl_idxs = None

			n_data_eff = dataset['F'].shape[0]
			if excl_idxs is not None:
				n_data_eff -= len(excl_idxs)

			if n_test is None and n_data_eff != 0: # test on all data points that have not been used for training or testing
				n_test = n_data_eff
				print ui.info_str('[INFO]') + ' Test set size was automatically set to %d points.' % n_test

			if n_test == 0 or n_data_eff == 0:
				print ui.warn_str('[WARN]') + ' Skipping! No unused points for test in provided dataset.\n'
				return
			elif n_data_eff < n_test:
				n_test = n_data_eff
				print ui.warn_str('[WARN]') + ' Test size reduced to %d. Not enough unused points in provided dataset.\n' % n_test

			if 'E' in dataset:
				test_idxs = gdml.draw_strat_sample(dataset['E'], n_test, excl_idxs=excl_idxs)
			else:
				print ui.warn_str('[WARN]') + ' Test dataset will be sampled with no guidance from energy labels (uniformly)!'
				print '       Note: Larger test datasets are recommended due to slower convergence of the error.'
				idxs_valid = np.random.choice(np.arange(n_valid), n_valid, replace=False)
		np.random.shuffle(test_idxs) # shuffle to improve convergence of online error

		z = dataset['z']
		R = dataset['R'][test_idxs,:,:]
		F = dataset['F'][test_idxs,:,:]

		if model['use_E']:
			E = dataset['E'][test_idxs]

		gdml_predict = GDMLPredict(model, max_processes=max_processes)

		if num_workers == 0 or batch_size == 0:
			sys.stdout.write('\r[' + ui.blink_str(' .. ') + '] Running benchmark...')
			sys.stdout.flush()

			#n_reps = min(max(int(n_valid*0.1),3),10) # 10% of the n_valid, at least 3, no more than 5
			n_reps = 3
			gps = gdml_predict.set_opt_num_workers_and_batch_size_fast(n_bulk=100,n_reps=n_reps)
			num_workers, batch_size, bulk_mp = gdml_predict._num_workers, gdml_predict._chunk_size, gdml_predict._bulk_mp

			sys.stdout.write('\r[DONE] Running benchmark... ' + ui.gray_str('(best: %d wkr %s/ chunks of %d @ %.1f geo/s)\n' % (num_workers, '[MP] ' if bulk_mp else '', batch_size, gps)))
			sys.stdout.flush()
		else:
			gdml_predict.set_num_workers(num_workers)
			gdml_predict.set_batch_size(batch_size)
			gdml_predict._bulk_mp = bulk_mp # TODO: implement setter 

		n_atoms = z.shape[0]

		if model['use_E']:
			e_mae_sum, e_rmse_sum = 0,0
		f_mae_sum, f_rmse_sum = 0,0
		cos_mae_sum, cos_rmse_sum = 0,0
		mag_mae_sum, mag_rmse_sum = 0,0

		b_size = min(100, len(test_idxs))
		n_done = 0
		t = time.time()
		for b_range in _batch(range(len(test_idxs)), b_size):

			n_done_step = len(b_range)
			n_done += n_done_step

			r = R[b_range].reshape(n_done_step,-1)
			e_pred,f_pred = gdml_predict.predict(r)

			if model['use_E']:
				e = E[b_range]
			f = F[b_range].reshape(n_done_step,-1)

			# energy error
			if model['use_E']:
				e_mae, e_mae_sum, e_rmse, e_rmse_sum = _online_err(np.squeeze(e) - e_pred, 1, n_done, e_mae_sum, e_rmse_sum)

			# force component error
			f_mae, f_mae_sum, f_rmse, f_rmse_sum = _online_err(f - f_pred, 3*n_atoms, n_done, f_mae_sum, f_rmse_sum)

			# magnetude error
			f_pred_mags = np.linalg.norm(f_pred.reshape(-1,3), axis=1)
			f_mags = np.linalg.norm(f.reshape(-1,3), axis=1)
			mag_mae, mag_mae_sum, mag_rmse, mag_rmse_sum = _online_err(f_pred_mags - f_mags, n_atoms, n_done, mag_mae_sum, mag_rmse_sum)

			# normalized cosine error
			f_pred_norm = f_pred.reshape(-1,3) / f_pred_mags[:,None]
			f_norm = f.reshape(-1,3) / f_mags[:,None]
			cos_err = np.arccos(np.einsum('ij,ij->i', f_pred_norm, f_norm)) / np.pi
			cos_mae, cos_mae_sum, cos_rmse, cos_rmse_sum = _online_err(cos_err, n_atoms, n_done, cos_mae_sum, cos_rmse_sum)

			progr = float(n_done) / len(test_idxs)
			sps = n_done / (time.time() - t) # samples per second (don't time the benchmark predictions)
			
			sys.stdout.write('\r[%3d%%] >> ' % (progr * 100))
			if model['use_E']:
				sys.stdout.write('Energy: %.3f/%.3f - ' % (e_mae,e_rmse))
			sys.stdout.write('Forces: %.3f/%.3f (MAE, RMSE)' % (f_mae,f_rmse))
			#sys.stdout.write('\r[%3d%%] >> Energy: %.3f/%.3f - Forces: %.3f/%.3f (MAE, RMSE)' % (progr * 100,e_mae,e_rmse,f_mae,f_rmse))
			if b_range is not None:
				sys.stdout.write(ui.gray_str(' @ %.1f geo/s' % sps))
			sys.stdout.flush()
		print '\n'

		if model['use_E']:
			e_rmse_pct = ((e_rmse/e_err['rmse'] - 1.) * 100)
		f_rmse_pct = ((f_rmse/f_err['rmse'] - 1.) * 100)

		if func_called_directly and n_models == 1:
			print ui.white_bold_str('Measured errors (MAE, RMSE):')
			format_str = ' {:<16} {:>.2e}/{:>.2e} '
			if model['use_E']:
				print (format_str + '[a.u.] {:<}').format('Energy:', e_mae, e_rmse, "%s (%+.1f %%)" % ('OK' if e_mae <= e_err['mae'] and e_rmse <= e_err['rmse'] else '!!',e_rmse_pct))
			print (format_str + '[a.u.] {:<}').format('Forces:', f_mae, f_rmse, "%s (%+.1f %%)" % ('OK' if f_mae <= f_err['mae'] and f_rmse <= f_err['rmse'] else '!!',f_rmse_pct))
			print (format_str + '[a.u.]').format(' Magnitude:', mag_mae, mag_rmse)
			print (format_str + '[0-1], 0: best').format(' Angle:', cos_mae, cos_rmse)
			print

		model_mutable = dict(model)
		model.close()
		model = model_mutable

		model_needs_update = overwrite or needs_valid or model['n_test'] < len(test_idxs)
		if model_needs_update:
			if is_test:
				model['n_test'] = len(test_idxs)
				model['md5_test'] = dataset['md5']
			
			if model['use_E']:
				model['e_err'] = {'mae':np.asscalar(e_mae),'rmse':np.asscalar(e_rmse)}
			model['f_err'] = {'mae':np.asscalar(f_mae),'rmse':np.asscalar(f_rmse)}
			np.savez_compressed(model_path, **model)

		if is_test:
			if overwrite:
				print ui.info_str('[INFO]') + ' Errors were updated in model file.\n'
			elif len(test_idxs) < model['n_test']: # test on less than the model has been previously tested on
				model_path = os.path.join(model_dir, model_file_names[i])
				print ui.warn_str('[WARN]') + ' Model has previously been tested on %d points. Errors for current run with %d points have NOT been recorded in model file.' % (model['n_test'], len(test_idxs)) +\
								  '\n       Run \'%s test -o %s %s %s\' to overwrite.\n' % (PACKAGE_NAME, os.path.relpath(model_path), dataset_path, n_test)

def select(model_dir, overwrite, max_processes, command=None, **kwargs):

	func_called_directly = command == 'select' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n MODEL SELECTION \n') + '-'*100

	model_dir, model_file_names = model_dir

	use_E = True

	rows = []
	data_names = ['sig', 'MAE', 'RMSE', 'MAE', 'RMSE']
	for i,model_file_name in enumerate(model_file_names):
		model_path = os.path.join(model_dir, model_file_name)
		_, model = ui.is_file_type(model_path, 'model')

		use_E = model['use_E']

		if i == 0:
			idxs_train = set(model['idxs_train'])
			md5_train = model['md5_train']
			idxs_valid = set(model['idxs_valid'])
			md5_valid = model['md5_valid']
		else:
			if (md5_train != model['md5_train']
			or md5_valid != model['md5_valid']
			or idxs_train != set(model['idxs_train'])
			or idxs_valid != set(model['idxs_valid'])):
				raise AssistantError('{} contains models trained or validated on different datasets.'.format(model_dir))

		e_err = {'mae': 0., 'rmse': 0.}
		if model['use_E']:
			e_err = model['e_err'].item()
		f_err = model['f_err'].item()

		rows.append([model['sig'],\
					 e_err['mae'],\
					 e_err['rmse'],\
					 f_err['mae'],\
					 f_err['rmse']])

		model.close()

	f_rmse_col = [row[4] for row in rows]
	best_idx = f_rmse_col.index(min(f_rmse_col)) # idx of row with lowest f_rmse
	best_sig = rows[best_idx][0]

	rows = sorted(rows, key=lambda col: col[0]) # sort according to sigma
	print ui.white_bold_str('Cross-validation errors')
	print ' '*7 + 'Energy' + ' '*6 + 'Forces'
	print (' {:>3} ' + '{:>5} '*4).format(*data_names)
	print ' ' + '-'*27
	format_str = ' {:>3} ' + '{:>5.2f} '*4
	format_str_no_E = ' {:>3}     -     - ' + '{:>5.2f} '*2
	for row in rows:
		if use_E:
			row_str = format_str.format(*row)
		else:
			row_str = format_str_no_E.format(*[row[0], row[3], row[4]])

		if row[0] != best_sig:
			row_str = ui.gray_str(row_str)
		print row_str
	print

	has_printed = False

	sig_col = [row[0] for row in rows]
	if best_sig == min(sig_col) or best_sig == max(sig_col):
		print ui.warn_str('[WARN]') + ' Optimal sigma lies on boundary of search grid.' +\
						  '\n       Model performance might improve if search grid is extended in direction sigma %s %d.' % ('<' if best_idx == 0 else '>', best_sig)
		has_printed = True

	best_model_file_name = model_file_names[best_idx]
	best_model = np.load(os.path.join(model_dir, best_model_file_name))
	best_model_file_name_extended = io.model_file_name(best_model, is_extended=True)

	model_exists = os.path.isfile(best_model_file_name_extended)
	if model_exists and overwrite:
		print ui.info_str('[INFO]') + ' Overwriting existing model file.'
		has_printed = True
	if not model_exists or overwrite:
		if func_called_directly:
			print '[DONE] Writing model file \'%s\'...' % best_model_file_name_extended
			has_printed = True
		shutil.copy(os.path.join(model_dir, best_model_file_name), best_model_file_name_extended)
		shutil.rmtree(model_dir)
	else:
		print ui.warn_str('[WARN]') + ' Model \'%s\' already exists.' % best_model_file_name_extended +\
									  '\n       Run \'%s select -o %s\' to overwrite.' % (PACKAGE_NAME, os.path.relpath(model_dir))
		has_printed = True
	
	if has_printed: # Add extra newline.
		print

	if func_called_directly:
		print ui.white_back_str(' NEXT STEP ') + ' %s test %s %s %s\n' % (PACKAGE_NAME, best_model_file_name_extended, '<dataset_file>', '<n_test>')

	best_model.close()
	return best_model_file_name_extended


def show(file, overwrite, max_processes, command=None, **kwargs):

	print ui.white_back_str('\n SHOW DETAILS \n') + '-'*100
	file_path, file = file

	if file['type'] == 'd':
		print ui.white_bold_str('Dataset properties')
		_print_dataset_properties(file)

	if file['type'] == 't':
		print ui.white_bold_str('Task properties')
		_print_task_properties(use_sym=file['use_sym'], use_cprsn=file['use_cprsn'], use_E=file['use_E'], use_E_cstr=file['use_E_cstr'])

	if file['type'] == 'm':
		print ui.white_bold_str('Model properties')
		_print_model_properties(file)


def main():

	def _add_argument_dataset(parser, help='path to dataset file'):
		parser.add_argument('dataset',\
							metavar 	= '<dataset_file>',\
							type    	= lambda x: ui.is_file_type(x, 'dataset'),\
							help 		= help)

	def _add_argument_sample_size(parser, subset_str):
		subparser.add_argument('n_%s' % subset_str,\
								metavar	= '<n_%s>' % subset_str,\
								type 	= ui.is_strict_pos_int,\
								help	= '%s sample size' % subset_str)

	def _add_argument_dir_with_file_type(parser, type, or_file=False):
		parser.add_argument('%s_dir' % type,\
							metavar 	= '<%s_dir%s>' % (type, '_or_file' if or_file else ''),\
							type 		= lambda x: ui.is_dir_with_file_type(x, type, or_file=or_file),\
							help 		= 'path to %s directory%s' % (type, ' or file' if or_file else ''))

	parser = argparse.ArgumentParser()
	parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)     
	
	parent_parser 	= argparse.ArgumentParser(add_help=False)                  
	parent_parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help='overwrite existing files')
	parent_parser.add_argument('-p', '--max_processes',\
								metavar = '<max_processes>',\
								type 	= ui.is_strict_pos_int,\
								help 	= 'limit the number of processes for this application')
	
	subparsers 		= parser.add_subparsers(title='commands', dest='command')
	parser_all 		= subparsers.add_parser('all', help='create model from beginning to end', parents = [parent_parser])
	parser_create 	= subparsers.add_parser('create', help='create training task(s)', parents = [parent_parser])
	parser_train 	= subparsers.add_parser('train', help='train model(s) from task(s)', parents = [parent_parser])
	parser_valid 	= subparsers.add_parser('validate', help='validate model(s)', parents = [parent_parser])
	parser_select 	= subparsers.add_parser('select', help='select best performing model', parents = [parent_parser])
	parser_test 	= subparsers.add_parser('test', help='test a model', parents = [parent_parser])
	parser_show 	= subparsers.add_parser('show', help='print details about dataset, task or model file', parents = [parent_parser])

	for subparser in [parser_all, parser_create]:
		_add_argument_dataset(subparser, help='path to dataset file (train/validation/test subsets are sampled from here if no seperate dataset are specified)')
		_add_argument_sample_size(subparser, 'train')
		_add_argument_sample_size(subparser, 'valid')
		subparser.add_argument('-v', '--validation_dataset',\
								metavar = '<validation_dataset_file>',\
								dest 	= 'valid_dataset',\
								type    = lambda x: ui.is_file_type(x, 'dataset'),\
								help 	= 'path to validation dataset file')
		subparser.add_argument('-t', '--test_dataset',\
								metavar = '<test_dataset_file>',\
								dest 	= 'test_dataset',\
								type    = lambda x: ui.is_file_type(x, 'dataset'),\
								help 	= 'path to test dataset file')
		subparser.add_argument('-s', '--sig',\
								metavar = ('<s1>', '<s2>'),\
								dest 	= 'sigs',\
								type 	= ui.parse_list_or_range,\
								help 	= 'integer list and/or range <start>:[<step>:]<stop> for the kernel hyper-parameter sigma',\
								nargs 	= '+')
		
		group = subparser.add_mutually_exclusive_group()
		group.add_argument('--gdml',\
								action 	= 'store_true',\
								help 	= 'don\'t include symmetries in the model (GDML)')
		group.add_argument('--cprsn',\
								dest 	= 'use_cprsn',\
								action 	= 'store_true',\
								help 	= 'compress kernel matrix along symmetric degrees of freedom')

		group = subparser.add_mutually_exclusive_group()
		group.add_argument('--no_E',\
								dest 	= 'use_E',\
								action 	= 'store_false',\
								help 	= 'only reconstruct force field w/o potential energy surface')
		group.add_argument('--E_cstr',\
								dest 	= 'use_E_cstr',\
								action 	= 'store_true',\
								help 	= 'include the energy constraints in the kernel')

	for subparser in [parser_valid, parser_test]:
		_add_argument_dir_with_file_type(subparser, 'model', or_file=True)
		_add_argument_dataset(subparser)

	for subparser in [parser_all, parser_test]:
		subparser.add_argument('n_test',
								metavar	= '<n_test>',\
								type	= ui.is_strict_pos_int,\
								help	= 'test sample size',\
								nargs 	= '?',\
								default = None)

	# train
	_add_argument_dir_with_file_type(parser_train, 'task', or_file=True)

	# select
	_add_argument_dir_with_file_type(parser_select, 'model')

	# show
	parser_show.add_argument('file',\
							metavar 	= '<file>',\
							type    	= lambda x: ui.is_valid_file_type(x),\
							help 		= help)


	args = parser.parse_args()

	# post-processing for optional sig argument
	if 'sigs' in args and args.sigs is not None:
		args.sigs = np.hstack(args.sigs).tolist() # flatten list, if (part of it) was generated using the range syntax
		args.sigs = sorted(list(set(args.sigs))) # remove potential duplicates

	_print_splash()

	try:
		getattr(sys.modules[__name__], args.command)(**vars(args))
	except AssistantError, err:
		sys.exit(ui.fail_str('[FAIL]') + ' %s' % err)


if __name__== "__main__":
	main()
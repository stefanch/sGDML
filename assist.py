#!/usr/bin/python

# GDML Force Field
# Author: Stefan Chmiela (stefan@chmiela.com)

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import time
import shutil
import re # task filename generation

import numpy as np

from src.gdml_train import GDMLTrain
from src.gdml_predict import GDMLPredict

from src.utils import io,ui

MODEL_DIR = BASE_DIR + '/models/assist/'


def _print_dataset_properties(dataset):

	n_mols, n_atoms, _ = dataset['R'].shape
	print ' {:<14} {:<} ({:<d} atoms)'.format('Name:', dataset['name'], n_atoms)
	print ' {:<14} {:<}'.format('Theory:', dataset['theory'])
	print ' {:<14} {:<d}'.format('# Points:', n_mols)

	T_min, T_max = np.min(dataset['E']), np.max(dataset['E'])
	print ' {:<14} {:<.3} '.format('Energies:', T_min) + '|--' + ' {:^8.3} '.format(T_max-T_min) + '--|' + ' {:>9.3} [a.u.]'.format(T_max)

	TG_min, TG_max = np.min(dataset['F'].ravel()), np.max(dataset['F'].ravel())
	print ' {:<14} {:<.3} '.format('Forces:', TG_min) + '|--' + ' {:.3} '.format(TG_max-TG_min) + '--|' + ' {:>9.3} [a.u.]'.format(TG_max)

	print ' {:<14} {:<}'.format('Fingerprint:', dataset['md5'])
	print

def _print_model_properties(model):
	
	print ' {:<14} {:<d}'.format('# Symmetries', len(model['perms']))

	n_train = len(model['train_idxs'])
	print ' {:<14} {:<d} points from \'{:<}\''.format('Training:', n_train, model['train_md5'])

	e_err = model['e_err'].item()
	f_err = model['f_err'].item()

	n_test = len(model['test_idxs'])
	is_tested = not np.isnan(e_err['mae']) and not np.isnan(e_err['rmse']) and not np.isnan(f_err['mae']) and not np.isnan(f_err['rmse'])
	print ' {:<14} {}{:<d} points from \'{:<}\''.format('Testing:', '' if is_tested else '[pending] ', n_test, model['test_md5'])

	n_valid = int(model['n_valid'])
	is_valid = n_valid > 0
	#if is_valid:
	#	print ' {:<14} {:<d} points from \'{:<}\''.format('Validation:', n_valid, model['valid_md5'])
	#else:
	#print ' {:<14} {}'.format('Validation:', '[pending]')
	print ' {:<14} {}'.format('Validation:', '{} points'.format(n_valid) if is_valid else '[pending]')

	action_str = 'Test' if not is_tested else 'Expected validation'
	print ' {:<14}'.format('{} errors:'.format(action_str))
	print '  {:<13} {:>.2e}/{:>.2e} [a.u.]'.format('Energy', e_err['mae'], e_err['rmse'])
	print '  {:<13} {:>.2e}/{:>.2e} [a.u.]'.format('Forces', f_err['mae'], f_err['rmse'])
	print

#all
def all(dataset, test_dataset, n_train, n_test, n_valid, gdml, overwrite, **kwargs):

	print '\n' + '-'*100
	print 'sGDML Model Creation Assistant'
	print '-'*100

	print '\n' + ui.white_back_str(' STEP 0 ') + ' Dataset(s)'
	print '-'*100

	print ui.white_bold_str('Properties')
	_, dataset_extracted = dataset
	_print_dataset_properties(dataset_extracted)

	if test_dataset is not None:
		print ui.underline_str('Properties (test)')
		_, test_dataset_extracted = test_dataset
		_print_dataset_properties(test_dataset_extracted)

	print ui.white_back_str(' STEP 1 ') + ' Create cross-validation tasks.'
	print '-'*100
	task_dir = create(dataset, test_dataset, n_train, n_test, gdml, overwrite, **kwargs)

	print ui.white_back_str(' STEP 2 ') + ' Train all models.'
	print '-'*100
	task_dir_arg = ui.is_dir_with_file_type(task_dir, 'task')
	model_dir = train(task_dir_arg, overwrite, **kwargs)

	print ui.white_back_str(' STEP 3 ') + ' Test all models.'
	print '-'*100
	model_dir_arg = ui.is_dir_with_file_type(model_dir, 'model')
	test(model_dir_arg, dataset, overwrite=False, **kwargs)

	print ui.white_back_str(' STEP 4 ') + ' Select best hyper-parameter combination.'
	print '-'*100
	best_model_path = select(model_dir_arg, overwrite, **kwargs)

	print ui.white_back_str(' STEP 5 ') + ' Validate selected model.'
	print '-'*100
	model_dir_arg = ui.is_dir_with_file_type(best_model_path, 'model', or_file=True)
	validate(model_dir_arg, dataset, n_valid, overwrite=False, **kwargs)

	print ui.pass_str('[DONE]') + ' Training assistant finished sucessfully.'
	print '       Find your model here: \'%s\'' % os.path.relpath(best_model_path, BASE_DIR)

# create tasks
def create(dataset, test_dataset, n_train, n_test, gdml, overwrite, command=None, **kwargs):

	dataset_path, dataset = dataset
	n_data = dataset['E'].shape[0]

	func_called_directly = command == 'create' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n TASK CREATION ')
		print '-'*100

		print ui.white_bold_str('Dataset properties')
		_print_dataset_properties(dataset)

	if n_data < n_train:
		raise ValueError('dataset only contains {} points, can not train on {}'.format(n_data,n_train))
	elif test_dataset is None and n_data - n_train < n_test:
		raise ValueError('dataset only contains {} points, can not train on {} and test on'.format(n_data,n_train,n_test))
	elif test_dataset is not None and test_dataset['E'].shape[0] < n_test:
		raise ValueError('test dataset only contains {} points, can not test on {}'.format(n_data,n_test))

	if test_dataset == None:
		test_dataset_path = dataset_path
		test_dataset = dataset

	recov_sym = not gdml

	gdml = GDMLTrain()

	theory_level_str = re.sub('[^\w\-_\.]', '_', str(dataset['theory']))
	theory_level_str = re.sub('__', '_', theory_level_str)
	dataset_name_str = str(dataset['name'])
	task_dir = BASE_DIR + '/training/' + dataset_name_str + '-' + theory_level_str + '-' + str(n_train)
	task_reldir = os.path.relpath(task_dir, BASE_DIR)

	if os.path.exists(task_dir):
		if args.overwrite:
			print ui.info_str('[INFO]') + ' Overwriting existing training directory.'
			for f in os.listdir(task_dir):
				os.remove(os.path.join(task_dir, f))
		else:
			print ui.warn_str('[WARN]') + ' Skipping existing task dir \'%s\'.\n' % task_reldir
			if func_called_directly:
				print '       Run \'python %s -o create %s %d %d\' to overwrite.\n' % (os.path.basename(__file__), dataset_path, n_train, n_test)
			return task_dir
	else:
		os.makedirs(task_dir)

	lam = 1e-15
	sig_range = range(2,100,10)

	task = gdml.create_task(dataset, n_train, test_dataset, n_test, sig=1, lam=lam, recov_sym=recov_sym)

	print '[DONE] Writing %d tasks with %s training points each.' % (len(sig_range), task['R_train'].shape[0])
	for sig in sig_range:
		task['sig'] = sig

		task_path = task_dir + '/task-' + io.task_file_name(task)
		if os.path.isfile(task_path + '.npz'):
			print ui.info_str('[INFO]') + ' Skipping exising task \'task-' + io.task_file_name(task) + '.npz\'.'
		else:
			np.savez_compressed(task_path, **task)
	print ''

	if func_called_directly:
		print ui.white_back_str(' NEXT STEP ') + ' python %s train %s\n' % (os.path.basename(__file__), task_reldir)

	return task_dir


# train models
def train(task_dir, overwrite, command=None, **kwargs):

	task_dir, task_files = task_dir

	func_called_directly = command == 'train' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n TRAINING ')
		print '-'*100

	print_verbose = len(task_files) == 1

	gdml = GDMLTrain()
	for i,task_file in enumerate(task_files):

		if print_verbose:
			print ui.white_bold_str('Training task')
		else:
			print ui.white_bold_str('Training task %d of %d' % (i+1,len(task_files)))

		task_path = os.path.join(task_dir, task_file)
		_, task = ui.is_file_type(task_path, 'task')

		model_path = os.path.join(task_dir, 'model-' + io.task_file_name(task))
		if not overwrite and os.path.isfile(model_path):
			print ui.warn_str('[WARN]') + ' Skipping exising model \'model-' + io.task_file_name(task) + '\'.'
			if func_called_directly:
				print '       Run \'python %s -o train %s\' to overwrite.' % (os.path.basename(__file__), task_dir)
			print
			continue

		model = gdml.train(task)
		model['c'] = gdml.recov_int_const(model, task)

		if print_verbose:
			print '[DONE] Writing model to file \'%s\'...\n' % os.path.relpath(model_path, BASE_DIR)
		np.savez_compressed(model_path, **model)
		print

	if func_called_directly:
		print ui.white_back_str(' NEXT STEP ') + ' python %s test %s %s\n' % (os.path.basename(__file__), task_dir, '<dataset_file>')

	return task_dir #model directory

def _batch(iterable, n=1,first_none=False):	
	l = len(iterable)
	if first_none:
		yield None
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]

def _online_err(err, size, n, mae_n_sum, rmse_n_sum):

	err = np.abs(err)

	mae_n_sum += np.sum(err) / size
	mae = mae_n_sum / n

	rmse_n_sum += np.sum(err**2) / size
	rmse = np.sqrt(rmse_n_sum / n)

	return mae, mae_n_sum, rmse, rmse_n_sum


def test(model_dir, dataset, overwrite, command=None, **kwargs):

	dataset_path_extracted, dataset_extracted = dataset

	func_called_directly = command == 'test' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n TESTING ')
		print '-'*100

		print ui.white_bold_str('Dataset properties')
		_print_dataset_properties(dataset_extracted)

	n_valid = 0
	validate(model_dir, dataset, n_valid, overwrite, command, **kwargs)

	if func_called_directly:
		model_dir_extracted, model_files = model_dir
		if len(model_files) == 1:
			model_path = os.path.join(model_dir_extracted, model_files[0])
			model_relpath = os.path.relpath(model_path, BASE_DIR)
			print ui.white_back_str(' NEXT STEP ') + ' python %s validate %s %s %s\n' % (os.path.basename(__file__), model_relpath, dataset_path_extracted, '<n_validate>')
		else:
			print ui.white_back_str(' NEXT STEP ') + ' python %s select %s\n' % (os.path.basename(__file__), model_dir_extracted)


def validate(model_dir, dataset, n_valid, overwrite, command=None, **kwargs):

	model_dir, model_files = model_dir
	dataset_path, dataset = dataset

	func_called_directly = command == 'validate' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n VALIDATION ')
		print '-'*100

		print ui.white_bold_str('Dataset properties')
		_print_dataset_properties(dataset)

	num_workers, batch_size = 0,0
	for i,model_file in enumerate(model_files):

		model_path = os.path.join(model_dir, model_file)
		_, model = ui.is_file_type(model_path, 'model')

		if len(model_files) == 1:
			print ui.white_bold_str('Model properties')
			_print_model_properties(model)

		e_err = model['e_err'].item()
		f_err = model['f_err'].item()
	
		# is this a test or validation run?
		needs_test = np.isnan(e_err['mae']) and np.isnan(e_err['rmse']) and np.isnan(f_err['mae']) and np.isnan(f_err['rmse'])
		is_valid = n_valid != 0 and not needs_test

		action_str = 'Validating' if is_valid else 'Testing'
		if len(model_files) == 1:
			print ui.white_bold_str('%s model' % action_str)
		else:
			print ui.white_bold_str('%s model %d of %d' % (action_str, i+1, len(model_files)))

		if not overwrite and not needs_test and not is_valid:
			model_relpath = os.path.relpath(model_path, BASE_DIR)
			print ui.warn_str('[WARN]') + ' Skipping already tested model \'%s\'.' % model_relpath
			if command == 'test':
				print '       Run \'python %s -o test %s %s\' to overwrite.' % (os.path.basename(__file__), model_path, dataset_path)
			print
			continue

		# TODO:
		# (1) check if user tried to validate an untested model
		# (2) if test, check if the correct test dataset was provided (done?)
		# (3) 

		if needs_test and dataset['md5'] != model['test_md5']:
			raise OSError('fingerprint of provided test dataset does not match the one in model file.')

		valid_idxs = model['test_idxs']
		if is_valid:
			gdml = GDMLTrain()

			# exclude training and/or test sets from validation set if necessary
			excl_idxs = np.empty((0,), dtype=int)
			if dataset['md5'] == model['train_md5']:
				excl_idxs = np.concatenate([excl_idxs, model['train_idxs']])
			if dataset['md5'] == model['test_md5']:
				excl_idxs = np.concatenate([excl_idxs, model['test_idxs']])
			if len(excl_idxs) == 0:
				excl_idxs = None

			if n_valid is None: # test on all data points that have not been used for training or testing
				n_valid = len(dataset['E']) - len(excl_idxs)
				print ui.info_str('[INFO]') + ' Validation set size was automatically set to %d points.' % n_valid

			valid_idxs = gdml.draw_strat_sample(dataset['E'], n_valid, excl_idxs=excl_idxs)
		np.random.shuffle(valid_idxs) # shuffle to improve convergence of online error

		z = dataset['z']
		R_test = dataset['R'][valid_idxs,:,:]
		T_test = dataset['E'][valid_idxs]
		TG_test = dataset['F'][valid_idxs,:,:]

		gdml = GDMLPredict(model)

		n_bench = 0
		if num_workers == 0 or batch_size == 0:
			sys.stdout.write('\r[' + ui.blink_str(' .. ') + '] Running benchmark...')
			sys.stdout.flush()

			n_reps = 1000 if len(model_files) > 1 else 100 # do an extensive benchmark, if there is more than one model to test
			E_bench, F_bench = gdml.set_opt_num_workers_and_batch_size(R=R_test, n_reps=n_reps) #  the benchmark function takes uses real test data and returns part of the result for R_test
			n_bench = len(E_bench)
			num_workers, batch_size = gdml._num_workers, gdml._batch_size

			sys.stdout.write('\r[DONE] Running benchmark... ' + ui.gray_str('(%d workers w/ %d batch size)\n' % (num_workers, batch_size)))
			sys.stdout.flush()
		else:
			gdml.set_num_workers(num_workers)
			gdml.set_batch_size(batch_size)

		n_atoms = z.shape[0]

		e_mae_sum, e_rmse_sum = 0,0
		f_mae_sum, f_rmse_sum = 0,0
		cos_mae_sum, cos_rmse_sum = 0,0
		mag_mae_sum, mag_rmse_sum = 0,0

		b_size = 1
		n_done = 0
		t = time.time()
		for b_range in _batch(range(n_bench,len(valid_idxs)), b_size, first_none=n_bench!=0):

			if b_range is None: # first run
				n_done = n_bench

				e = T_test[:n_bench]
				f = TG_test[:n_bench].reshape(b_size,-1)

				E,F = E_bench, F_bench
			else:
				n_done += len(b_range)

				r = R_test[b_range].reshape(b_size,-1)
				e = T_test[b_range]
				f = TG_test[b_range].reshape(b_size,-1)

				E,F = gdml.predict(r)

			# energy error
			e_mae, e_mae_sum, e_rmse, e_rmse_sum = _online_err(np.squeeze(e) - E, 1, n_done, e_mae_sum, e_rmse_sum)

			# force component error
			f_mae, f_mae_sum, f_rmse, f_rmse_sum = _online_err(f.ravel() - F, 3*n_atoms, n_done, f_mae_sum, f_rmse_sum)

			# magnetude error
			F_mags = np.linalg.norm(F.reshape(-1,3), axis=1)
			f_mags = np.linalg.norm(f.reshape(-1,3), axis=1)
			mag_mae, mag_mae_sum, mag_rmse, mag_rmse_sum = _online_err(F_mags - f_mags, n_atoms, n_done, mag_mae_sum, mag_rmse_sum)

			# normalized cosine error
			F_norm = F.reshape(-1,3) / F_mags[:,None]
			f_norm = f.reshape(-1,3) / f_mags[:,None]
			cos_err = np.arccos(np.einsum('ij,ij->i', F_norm, f_norm)) / np.pi
			cos_mae, cos_mae_sum, cos_rmse, cos_rmse_sum = _online_err(cos_err, n_atoms, n_done, cos_mae_sum, cos_rmse_sum)

			progr = float(n_done) / len(valid_idxs)
			sps = (n_done - n_bench) / (time.time() - t) # samples per second (don't time the benchmark predictions)
			
			sys.stdout.write('\r[%3d%%] >> Energy: %.3f/%.3f - Forces: %.3f/%.3f (MAE, RMSE)' % (progr * 100,e_mae,e_rmse,f_mae,f_rmse))
			if b_range is not None:
				sys.stdout.write(ui.gray_str(' @ %.1f geo/s' % sps))
			sys.stdout.flush()
		print

		e_rmse_pct = ((e_rmse/e_err['rmse'] - 1.) * 100)
		f_rmse_pct = ((f_rmse/f_err['rmse'] - 1.) * 100)

		if func_called_directly and len(model_files) == 1:
			print ui.white_bold_str('\nMeasured errors (MAE, RMSE):')
			print ' {:<14} {:>.2e}/{:>.2e} [a.u.] {:<} '.format('Energy', e_mae, e_rmse, "%s (%+.1f %%)" % ('OK' if e_mae <= e_err['mae'] and e_rmse <= e_err['rmse'] else '!!',e_rmse_pct))
			print ' {:<14} {:>.2e}/{:>.2e} [a.u.] {:<} '.format('Forces', f_mae, f_rmse, "%s (%+.1f %%)" % ('OK' if f_mae <= f_err['mae'] and f_rmse <= f_err['rmse'] else '!!',f_rmse_pct))
			print ' {:<14} {:>.2e}/{:>.2e} [a.u.]'.format('Magnitude', mag_mae, mag_rmse)
			print ' {:<14} {:>.2e}/{:>.2e} [0-1], 0:best'.format('Angle', cos_mae, cos_rmse)
			print

		model_mutable = dict(model)
		model_needs_update = overwrite or needs_test or model['n_valid'] < len(valid_idxs)

		if is_valid:
			model_mutable['n_valid'] = len(valid_idxs)
			if overwrite:
				print ui.info_str('[INFO]') + ' Errors were updated in model file.\n'
			elif len(valid_idxs) <= model['n_valid']: # validating on less than the model has been previously validated on
				model_path = os.path.join(model_dir, model_files[i])
				model_relpath = os.path.relpath(model_path, BASE_DIR)
				print ui.warn_str('[WARN]') + ' Model has previously been validated on %d points. Errors for current run with just %d points have NOT been recorded in model file.' % (model['n_valid'], len(valid_idxs)) +\
								  '\n       Run \'python %s -o validate %s %s %s\' to overwrite.\n' % (os.path.basename(__file__), model_relpath, dataset_path, n_valid)
			else:
				print
		else:
			print

		if model_needs_update:
			model_mutable['e_err'] = {'mae':np.asscalar(e_mae),'rmse':np.asscalar(e_rmse)}
			model_mutable['f_err'] = {'mae':np.asscalar(f_mae),'rmse':np.asscalar(f_rmse)}
			np.savez_compressed(model_path, **model_mutable)


def select(model_dir, overwrite, command=None, **kwargs):

	func_called_directly = command == 'select' # has this function been called from command line or from 'all'?
	if func_called_directly:
		print ui.white_back_str('\n MODEL SELECTION ')
		print '-'*100

	model_dir, model_files = model_dir

	best_i = 0
	besf_f_rmse = np.inf

	rows = []
	data_names = ['sig', 'MAE', 'RMSE', 'MAE', 'RMSE']
	for i,model_file in enumerate(model_files):
		model = np.load(os.path.join(model_dir, model_file))

		if i == 0:
			train_idxs = set(model['train_idxs'])
			train_md5 = model['train_md5']
			test_idxs = set(model['test_idxs'])
			test_md5 = model['test_md5']
		else:
			if (train_md5 != model['train_md5']
			or test_md5 != model['test_md5']
			or train_idxs != set(model['train_idxs'])
			or test_idxs != set(model['test_idxs'])):
				raise OSError('{} contains models trained or tested on different datasets'.format(model_dir))

		e_err = model['e_err'].item()
		f_err = model['f_err'].item()

		if f_err['rmse'] < besf_f_rmse:
			best_i = i
			besf_f_rmse = f_err['rmse']

		rows.append([model['sig'],\
					 '%.2f' % e_err['mae'],\
					 '%.2f' % e_err['rmse'],\
					 '%.2f' % f_err['mae'],\
					 '%.2f' % f_err['rmse']])

	best_sig = rows[best_i][0]
	rows = sorted(rows, key=lambda col: col[0]) # sort according to sigma
	print ui.white_bold_str('Cross-validation errors')
	print ' '*7 + 'Energy' + ' '*6 + 'Forces'
	print (' {:>3} ' + '{:>5} '*4).format(*data_names)
	print ' ' + '-'*27
	row_format = ' {:>3} ' + '{:>5} '*4
	for row in rows:
		print row_format.format(*row) + ('*' if row[0] == best_sig else '')
	print ''

	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)

	best_model_file = model_files[best_i]
	best_model_target_path = os.path.join(MODEL_DIR, best_model_file)
	best_model_target_relpath = os.path.relpath(best_model_target_path, BASE_DIR)

	model_exists = os.path.isfile(best_model_target_path)
	if model_exists and overwrite:
		print ui.info_str('[INFO]') + ' Overwriting existing model file.'
	if not model_exists or overwrite:
		print '[DONE] Writing model file to \'%s\'...\n' % best_model_target_relpath
		shutil.copy(os.path.join(model_dir, best_model_file), best_model_target_path)
	else:
		print ui.warn_str('[WARN]') + ' Model \'%s\' already exists.' % best_model_target_relpath +\
									  '\n       Run \'python %s -o select %s\' to overwrite.\n' % (os.path.basename(__file__), model_dir)
	if func_called_directly:
		print ui.white_back_str(' NEXT STEP ') + ' python %s validate %s %s %s\n' % (os.path.basename(__file__), best_model_target_relpath, '<dataset_file>', '<n_validate>')

	return best_model_target_path


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing files')
	parser.add_argument('--version', action='version', version='%(prog)s [unknown]')
	subparsers = parser.add_subparsers(title='commands', dest='command')
	
	# all
	parser_all = subparsers.add_parser('all', help='create model from beginning to end')
	parser_all.add_argument('dataset', metavar = '<dataset_file>',\
										  type    = lambda x: ui.is_file_type(x, 'dataset'),\
										  help	   = 'path to dataset file (train and test data are both sampled from here, if no separate test set is specified)')
	parser_all.add_argument('n_train', metavar = '<n_train>',\
										  type    = lambda x: ui.is_strict_pos_int(x),\
										  help    = 'number of data points to use for training')
	parser_all.add_argument('n_test', metavar = '<n_test>',\
										 type    = lambda x: ui.is_strict_pos_int(x),\
										 help    = 'number of data points to use for testing')
	parser_all.add_argument('n_valid', metavar = '<n_validate>',\
										 type    = lambda x: ui.is_strict_pos_int(x),\
										 help    = 'number of data points to use for validation',\
										 nargs   = '?', default = None)
	parser_all.add_argument('--test_dataset', metavar = '<test_dataset_file>',\
												 type    = lambda x: ui.is_file_type(x, 'dataset'),\
												 help	   = 'path to test dataset file')
	parser_all.add_argument('--gdml', dest='gdml', action='store_true', help = 'don\'t include symmetries in the model')

	# create
	parser_create = subparsers.add_parser('create', help='create training task(s)')
	parser_create.add_argument('dataset', metavar = '<dataset_file>',\
										  type    = lambda x: ui.is_file_type(x, 'dataset'),\
										  help	   = 'path to dataset file (train and test data are both sampled from here, if no separate test set is specified)')
	parser_create.add_argument('n_train', metavar = '<n_train>',\
										  type    = lambda x: ui.is_strict_pos_int(x),\
										  help    = 'number of data points to use for training')
	parser_create.add_argument('n_test', metavar = '<n_test>',\
										 type    = lambda x: ui.is_strict_pos_int(x),\
										 help    = 'number of data points to use for testing')
	parser_create.add_argument('--test_dataset', metavar = '<test_dataset>',\
												 type    = lambda x: ui.is_file_type(x, 'dataset'),\
												 help	   = 'path to test dataset file')
	parser_create.add_argument('--gdml', dest='gdml', action='store_true', help = 'don\'t include symmetries in the model')
	
	# train
	parser_train = subparsers.add_parser('train', help='train model(s) from task(s)')
	parser_train.add_argument('task_dir', metavar = '<task_dir>',\
							 	type    = lambda x: ui.is_dir_with_file_type(x, 'task'),\
							 	help	 = 'path to task directory')

	# test
	parser_test = subparsers.add_parser('test', help='test model(s)')
	parser_test.add_argument('model_dir', metavar = '<model_dir>',\
								 	 type    = lambda x: ui.is_dir_with_file_type(x, 'model', or_file=True),\
								 	 help	 = 'path to model directory')
	parser_test.add_argument('dataset',   metavar = '<dataset_file>',\
								     type    = lambda x: ui.is_file_type(x, 'dataset'),\
								     help	= 'path to dataset file')

	# select
	parser_select = subparsers.add_parser('select', help='select best performing model')
	parser_select.add_argument('model_dir', metavar='<model_dir>',\
								 	 type    = lambda x: ui.is_dir_with_file_type(x, 'model'),\
								 	 help	 = 'path to model directory')

	# valid
	parser_valid = subparsers.add_parser('validate', help='validate a model')
	parser_valid.add_argument('model_dir', metavar='<model_dir>',\
								 	 type    = lambda x: ui.is_dir_with_file_type(x, 'model', or_file=True),\
								 	 help	 = 'path to model directory')
	parser_valid.add_argument('dataset',   metavar = '<dataset>',\
								     type    = lambda x: ui.is_file_type(x, 'dataset'),\
								     help	= 'path to dataset file')
	parser_valid.add_argument('n_valid', metavar = '<n_validate>',\
										 type    = lambda x: ui.is_strict_pos_int(x),\
										 help    = 'number of data points to use for validation',\
										 nargs   = '?', default = None)

	args = parser.parse_args()
	#print args

	getattr(sys.modules[__name__], args.command)(**vars(args))
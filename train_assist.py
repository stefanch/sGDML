#!/usr/bin/python

# GDML Force Field
# Author: Stefan Chmiela (stefan@chmiela.com)

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import re

import scipy.io
import numpy as np

from src.utils import io,ui

fail_msg = ui.fail_str('[FAIL]') + ' Training assistant failed.'


parser = argparse.ArgumentParser(description='Performs all steps necessary to train a sGDML model from a given dataset.')
parser.add_argument('dataset', metavar = '<[train_]dataset>',\
							   type    = lambda x: ui.is_file_type(x, 'dataset'),\
							   help	   = 'path to dataset file (train and test data are both sampled from here, if no separate test set is specified)')
parser.add_argument('n_train', metavar = '<n_train>',\
							   type    = lambda x: ui.is_strict_pos_int(x),\
							   help    = 'number of training points to sample from dataset')
parser.add_argument('n_test',  metavar = '<n_test>',\
							   type    = lambda x: ui.is_strict_pos_int(x),\
							   help    = 'number of points form dataset for testing')
parser.add_argument('n_valid', metavar = '<n_valid>',\
							   type    = lambda x: ui.is_strict_pos_int(x),\
							   help    = 'number of points form dataset for validation',\
							   nargs   = '?', default = None)
parser.add_argument('--test_dataset', metavar = '<test_dataset>', nargs='?',\
							   type    = lambda x: ui.is_file_type(x, 'dataset'),\
							   help	   = 'path to test dataset file')
parser.add_argument('--gdml', dest='gdml', action='store_true', help = 'don\'t include symmetries in the model')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing results')
args = parser.parse_args()
dataset_path, dataset = args.dataset


def print_dataset_properties(dataset):

	n_mols, n_atoms, _ = dataset['R'].shape

	print ' {:<14} {:<} ({:<d} atoms)'.format('Name:', dataset['name'], n_atoms)
	print ' {:<14} {:<}'.format('Theory:', dataset['theory'])
	print ' {:<14} {:<d}'.format('# Points:', n_mols)

	T_min, T_max = np.min(dataset['E']), np.max(dataset['E'])
	print ' {:<14} {:<.3} '.format('Energies', T_min) + '|--' + ' {:^8.3} '.format(T_max-T_min) + '--|' + ' {:>9.3} [a.u.]'.format(T_max)

	TG_min, TG_max = np.min(dataset['F'].ravel()), np.max(dataset['F'].ravel())
	print ' {:<14} {:<.3} '.format('Forces', TG_min) + '|--' + ' {:.3} '.format(TG_max-TG_min) + '--|' + ' {:>9.3} [a.u.]'.format(TG_max)

	print ' {:<14} {:<}'.format('Fingerprint:', io.dataset_md5(dataset))



#print ui.warn_str('[WARN]') + ' Geometries, forces and energies must have consistent units.'


print ''
print '-'*100
print 'sGDML Model Creation Assistant'
print '-'*100

print '\n' + ui.white_back_str(' STEP 0 ') + ' Dataset(s)'
print '-'*100

if args.test_dataset is not None:
#	print ui.info_str('[INFO]') + ' Separate test and train datasets specified.'
	test_dataset_path, test_dataset = args.test_dataset
else:
	test_dataset_path = dataset_path
	test_dataset = dataset

print ui.underline_str('Properties')
print_dataset_properties(dataset)
#print ' {:<14} {:<} ({:<d} atoms)'.format('Name:', dataset['name'], n_atoms)
#print ' {:<14} {:<}'.format('Theory:', dataset['theory'])
#print ' {:<14} {:<d}'.format('# Points:', n_mols)

#T_min, T_max = np.min(dataset['E']), np.max(dataset['E'])
#print ' {:<14} {:<.3} '.format('Energies', T_min) + '|--' + ' {:^8.3} '.format(T_max-T_min) + '--|' + ' {:>9.3} [a.u.]'.format(T_max)

#TG_min, TG_max = np.min(dataset['F'].ravel()), np.max(dataset['F'].ravel())
#print ' {:<14} {:<.3} '.format('Forces', TG_min) + '|--' + ' {:.3} '.format(TG_max-TG_min) + '--|' + ' {:>9.3} [a.u.]'.format(TG_max)

#print ' {:<14} {:<}'.format('Fingerprint:', io.dataset_md5(dataset))


if args.test_dataset is not None:
	print
	print ui.underline_str('Properties (TEST)')
	print_dataset_properties(test_dataset)


#if args.n_train > n_mols:
#	sys.exit(ui.fail_str('[FAIL]') + ' Training split too large for dataset size.')
#elif args.n_test > (n_mols - args.n_train):
#	sys.exit(ui.fail_str('[FAIL]') + ' Test split too large for dataset size (no duplicates allowed in splits).')
#elif args.n_valid and (args.n_valid > n_mols - args.n_train - args.n_test):
#	sys.exit(ui.fail_str('[FAIL]') + ' Validation split too large for dataset size (no duplicates allowed in splits).')

# err_code = os.system('python split_dataset.py ' + dataset_path + ' ' + str(args.n_train) + ' ' + str(args.n_test)\
#  				  + (' ' + str(args.n_valid) if args.n_valid else '')\
# 				  + (' --gdml' if args.gdml else '')\
# 				  + (' -o' if args.overwrite else ''))
# if err_code != 0: sys.exit(fail_msg)

# dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
# train_dataset_path = 'datasets/npz/splits/%s/train_%s.npz' % (dataset_name, dataset_name)
# test_dataset_path = 'datasets/npz/splits/%s/test_%s.npz' % (dataset_name, dataset_name)
# valid_dataset_path = 'datasets/npz/splits/%s/valid_%s.npz' % (dataset_name, dataset_name)


print '\n' + ui.white_back_str(' STEP 1 ') + ' Creating cross-validation tasks.'
print '-'*100
# TODO: change to train dataset
err_code = os.system('python cv_schedule.py ' + dataset_path + ' ' + str(args.n_train) + ' ' + str(args.n_test)\
				  + ' --test_dataset ' + test_dataset_path
				  + (' --gdml' if args.gdml else '')\
				  + (' -o' if args.overwrite else '')\
				  +  ' -s')
if err_code != 0: sys.exit(fail_msg)


print '\n' + ui.white_back_str(' STEP 2 ') + ' Training all models.'
print '-'*100
theory_level_str = re.sub('[^\w\-_\.]', '_', str(dataset['theory']))
theory_level_str = re.sub('__', '_', theory_level_str)
dataset_name_str = str(dataset['name'])
task_dir = 'training/' + dataset_name_str + '-' + theory_level_str + '-' + str(args.n_train)
err_code = os.system('python train_batch.py ' + task_dir\
				  + (' -o' if args.overwrite else '')\
				  +  ' -s')
if err_code != 0: sys.exit(fail_msg)


print '\n' + ui.white_back_str(' STEP 3 ') + ' Testing all models.'
print '-'*100
err_code = os.system('python test_batch.py ' + task_dir + ' ' + test_dataset_path + ' -s')
if err_code != 0: sys.exit(fail_msg)


print '\n' + ui.white_back_str(' STEP 4 ') + ' Select best hyper-parameter combination.'
print '-'*100
err_code = os.system('python cv_select.py ' + task_dir\
				  + (' -o' if args.overwrite else ''))
if err_code != 0: sys.exit(fail_msg)


#print '\n' + ui.white_back_str(' STEP 6 ') + ' Validating best model.'
#print '-'*100
#err_code = os.system('python test_batch.py ' + task_dir + ' ' + valid_dataset_path + ' ' + str(args.n_test)\
#				   + ' -s')
#if err_code != 0: sys.exit(fail_msg)

print ui.pass_str('[DONE]') + ' Training assistant finished sucessfully.'
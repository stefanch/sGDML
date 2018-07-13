#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import hashlib

import numpy as np

from src.gdml_train import GDMLTrain
from src.utils import ui

split_dataset_path = BASE_DIR + '/datasets/npz/splits/'


def split(dataset, idxs):
	subset = {}
	subset['R'] = dataset['R'][idxs,:,:]
	subset['T'] = dataset['T'][idxs,:]
	subset['TG'] = dataset['TG'][idxs,:,:]
	return subset
	

parser = argparse.ArgumentParser(description='Performs all steps necessary to train a sGDML model from a given dataset.')
parser.add_argument('dataset', metavar = '<dataset>',\
							   type    = lambda x: ui.is_valid_np_file(parser, x),\
							   help	   = 'path to bulk dataset file')
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
parser.add_argument('--gdml', dest='gdml', action='store_true', help = 'don\'t include symmetries in the model')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing results')
args = parser.parse_args()
dataset_path, dataset = args.dataset

n_mols, n_atoms, _ = dataset['R'].shape

print ui.underline_str('Dataset properties')
print ' {:<14} {:<} ({:<d} atoms)'.format('Name:', dataset['name'], n_atoms)
print ' {:<14} {:<}'.format('Theory level:', dataset['theory_level'])
print ' {:<14} {:<d}'.format('# Points:', n_mols)

T_min, T_max = np.min(dataset['T']), np.max(dataset['T'])
print ' {:<14} {:<.3} '.format('Energies', T_min) + u'\u251c\u2500' + ' {:^8.3} '.format(T_max-T_min) + u'\u2500\u2524' + ' {:>9.3} [UNIT]'.format(T_max)

TG_min, TG_max = np.min(dataset['TG'].ravel()), np.max(dataset['TG'].ravel())
print ' {:<14} {:<.3} '.format('Forces', T_min) + u'\u251c\u2500' + ' {:.3} '.format(TG_max-TG_min) + u'\u2500\u2524' + ' {:>9.3} [UNIT]'.format(TG_max)

md5_hash = hashlib.md5()
with open(dataset_path, 'rb') as f:
	for byte_block in iter(lambda: f.read(4096),b''): # read in chunks of 4K
		md5_hash.update(byte_block)
md5_str = md5_hash.hexdigest()
print ' {:<14} {:<}'.format('MD5:', md5_str)

if args.n_train > n_mols:
	sys.exit(ui.fail_str('[FAIL]') + ' Training split too large for dataset size.')
elif args.n_test > (n_mols - args.n_train):
	sys.exit(ui.fail_str('[FAIL]') + ' Test split too large for dataset size (no duplicates allowed in splits).')
elif args.n_valid and (args.n_valid > n_mols - args.n_train - args.n_test):
	sys.exit(ui.fail_str('[FAIL]') + ' Validation split too large for dataset size (no duplicates allowed in splits).')

if not os.path.exists(split_dataset_path):
	print ui.info_str(' [INFO]') + ' Created directory \'%s\'.' % 'datasets/npz/splits/'
	os.makedirs(split_dataset_path)

file_name = os.path.splitext(os.path.basename(dataset_path))[0]
if not os.path.exists(split_dataset_path + file_name):
	os.makedirs(split_dataset_path + file_name)
elif not args.overwrite:
	sys.exit(ui.fail_str('[FAIL]') + ' Dataset splits for \'' + file_name + '\' already exist.')


print ui.underline_str('\nSplits')

gdml = GDMLTrain()
dataset = dict(dataset)

train_idxs = gdml.draw_strat_sample(dataset['T'], args.n_train)
train_dataset = split(dataset, train_idxs)
print ' {:<14} {:<}'.format('Training', train_idxs.shape[0])

test_idxs = gdml.draw_strat_sample(dataset['T'], args.n_test, excl_idxs=train_idxs)
test_dataset = split(dataset, test_idxs)
print ' {:<14} {:<}'.format('Testing', test_idxs.shape[0])

n_valid = args.n_valid if args.n_valid else n_mols - args.n_train - args.n_test
if not args.n_valid:
	print ui.info_str(' [INFO]') + ' Validation dataset automatically set to %d.' % n_valid

valid_idxs = gdml.draw_strat_sample(dataset['T'], n_valid, excl_idxs=np.vstack((train_idxs[:,None],test_idxs[:,None])))
valid_dataset = split(dataset, valid_idxs)
print ' {:<14} {:<}'.format('Validation', valid_idxs.shape[0])

dataset['bulk_md5'] = md5_str
copy_keys = ['R', 'T', 'TG']

for key in copy_keys: dataset[key] = train_dataset[key]
dataset['bulk_idxs'] = train_idxs
train_dataset_path = split_dataset_path + file_name + '/train_' + file_name + '.npz'
np.savez_compressed(train_dataset_path, **dataset)

for key in copy_keys: dataset[key] = test_dataset[key]
dataset['bulk_idxs'] = test_idxs
test_dataset_path = split_dataset_path + file_name + '/test_' + file_name + '.npz'
np.savez_compressed(test_dataset_path, **dataset)

for key in copy_keys: dataset[key] = valid_dataset[key]
dataset['bulk_idxs'] = valid_idxs
valid_dataset_path = split_dataset_path + file_name + '/valid_' + file_name + '.npz'
np.savez_compressed(valid_dataset_path, **dataset)

print 'Dataset splits saved to \'datasets/npz/splits/' + file_name + '/\'\n'
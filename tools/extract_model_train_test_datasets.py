#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse
import numpy as np

from src.utils import io,ui


parser = argparse.ArgumentParser(description='Extracts the training and test data subsets used to construct a model.')
parser.add_argument('model', metavar = '<model_file>',\
							type = lambda x: ui.is_file_type(x, 'model'),\
							help = 'path to model file')
parser.add_argument('dataset', metavar = '<dataset_file>',\
							type = lambda x: ui.is_file_type(x, 'dataset'),\
							help = 'path to dataset file referenced in model')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing files')
args = parser.parse_args()

model_path, model = args.model
dataset_path, dataset = args.dataset


for s in ['train', 'test']:

	if dataset['md5'] != model[s + '_md5']:
		sys.exit(ui.fail_str('[FAIL]') + ' Dataset fingerprint does not match the one referenced in model for \'%s\'.' % s)

	idxs = model[s + '_idxs']
	R = dataset['R'][idxs,:,:]
	E = dataset['E'][idxs]
	F = dataset['F'][idxs,:,:]

	base_vars = {'type':		'd',\
				 'name':		dataset['name'],\
				 'theory':		dataset['theory'],\
				 'z':			dataset['z'],\
				 'R':			R,\
				 'E':			E,\
				 'F':			F}
	base_vars['md5'] = io.dataset_md5(base_vars)

	new_dataset_path = '%s_%s.npz' % (os.path.splitext(dataset_path)[0], s)
	new_dataset_relpath = os.path.relpath(new_dataset_path, BASE_DIR)

	file_exists = os.path.isfile(new_dataset_path)
	if file_exists and args.overwrite:
		print ui.info_str('[INFO]') + ' Overwriting existing model file.'
	if not file_exists or args.overwrite:
		np.savez_compressed(new_dataset_path, **base_vars)
		print '[DONE] Extracted %s dataset saved to \'%s\'' % (s,new_dataset_relpath)
	else:
		print ui.warn_str('[WARN]') + ' %s dataset \'%s\' already exists.' % (s.capitalize(), new_dataset_relpath) +\
									  '\n       Run \'python %s -o %s %s\' to overwrite.\n' % (os.path.basename(__file__), model_path, dataset_path)
		sys.exit()

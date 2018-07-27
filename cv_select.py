#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import shutil

import numpy as np

from src.utils import ui

best_model_target_dir = BASE_DIR + '/models/assist/'


parser = argparse.ArgumentParser(description='Retrieves best performing model for a given training set size.')
parser.add_argument('model_dir', metavar = '<model_dir>',\
							 	 type    = lambda x: ui.is_dir_with_file_type(x, 'model'),\
							 	 help	 = 'Path to model directory.')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing model')
args = parser.parse_args()

model_files, model_dir = args.model_dir

best_i = 0
besf_f_rmse = np.inf

data = []
data_names = ['sig', 'MAE', 'RMSE', 'MAE', 'RMSE']
for i,model_file in enumerate(model_files):
	try:
		model = np.load(os.path.join(model_dir, model_file))
	except:
		sys.exit("ERROR: Reading file failed.")

	if i == 0:
		print ' Name:          %s' % model['dataset_name']
		print ' Theory:        %s' % model['dataset_theory']
		print ' # Train:       %d' % len(model['train_idxs'])
		print ' # Test:        %d' % len(model['test_idxs'])

		train_idxs = set(model['train_idxs'])
		train_md5 = model['train_md5']
		test_idxs = set(model['test_idxs'])
		test_md5 = model['test_md5']
	else:
		if (train_md5 != model['train_md5']
		or test_md5 != model['test_md5']
		or train_idxs != set(model['train_idxs'])
		or test_idxs != set(model['test_idxs'])):
			sys.exit('error: %s contains models trained or tested on different datasets' % (model_dir))

	e_err = model['e_err'].item()
	f_err = model['f_err'].item()

	if f_err['rmse'] < besf_f_rmse:
		best_i = i
		besf_f_rmse = f_err['rmse']

	data.append([model['sig'],\
				 '%.2f' % e_err['mae'],\
				 '%.2f' % e_err['rmse'],\
				 '%.2f' % f_err['mae'],\
				 '%.2f' % f_err['rmse']])

best_sig = data[best_i][0]
data = sorted(data, key=lambda data_col: data_col[0]) # sort according to sigma
print '\n' + ui.underline_str('Cross-validation runs')
print ' '*6 + 'Energy' + ' '*6 + 'Forces'
print (' {:>3} ' + '{:>5} '*4).format(*data_names)
print ' ' + '-'*27
row_format = ' {:>3} ' + '{:>5} '*4
for row in data:
	print row_format.format(*row) + ('*' if row[0] == best_sig else '')
print ''

if not os.path.exists(best_model_target_dir):
	os.makedirs(best_model_target_dir)

best_model_file = model_files[best_i]
best_model_target_path = os.path.join(best_model_target_dir, best_model_file)

model_exists = os.path.isfile(best_model_target_path)
if model_exists and args.overwrite:	
	print ui.info_str('[INFO]') + ' Overwriting existing model file.'
if not model_exists or args.overwrite:
	print 'Writing model file to \'models/assist/%s\'...' % best_model_file
	shutil.copy(os.path.join(model_dir, best_model_file), best_model_target_path)
else:
	print ui.warn_str('[WARN]') + ' Model \'models/assist/%s\' already exists.' % best_model_file +\
								  '\n       Run \'cv_select.py -o %s\' to overwrite.' % model_dir
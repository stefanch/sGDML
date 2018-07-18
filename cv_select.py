#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import shutil

import numpy as np

from src.utils import ui

best_model_dir = BASE_DIR + '/models/assist/'


parser = argparse.ArgumentParser(description='Retrieves best performing model for a given training set size.')
parser.add_argument('model_dir', metavar = '<model_dir>',\
							 	 type    = lambda x: ui.is_dir(x),\
							 	 help	 = 'Path to model directory.')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing model')
args = parser.parse_args()


best_sig = 0
best_path = ''
best_sig_f_rmse = np.inf

data = []
data_names = ['Theory', '# train', '# test', 'sig', 'MAE', 'RMSE', 'MAE', 'RMSE']
name_str_maxlen = 0
theory_str_maxlen = len(data_names[1])
for file in sorted(os.listdir(args.model_dir)): # iterate in (lexicographic) order
	if file.endswith(".npz"):
		try:
			model = np.load(args.model_dir + '/' + file)
		except:
			sys.exit("ERROR: Reading file failed.")

		# Only process model files.
		if 'type' not in model or model['type'] != 'm':
			continue

		#theory_level_str = '?'
		#if 'theory_level' in model:
		#theory_str = str(model['dataset_theory'])

		n_test = np.nan
		#if 'n_test' in model:
		#	n_test = model['n_test']

		e_err = model['e_err'].item()
		f_err = model['f_err'].item()

		if f_err['rmse'] < best_sig_f_rmse:
			best_sig_f_rmse = f_err['rmse']
			best_sig = model['sig']
			best_path = file

		name_str_maxlen = max(name_str_maxlen,len(str(model['dataset_name'])))
		theory_str_maxlen = max(theory_str_maxlen,len(str(model['dataset_theory'])))
		data.append([model['dataset_name'],\
					 model['dataset_theory'],\
					 str(model['R_desc'].shape[1]),\
					 n_test,\
					 model['sig'],\
					 '%.2f' % e_err['mae'],\
					 '%.2f' % e_err['rmse'],\
					 '%.2f' % f_err['mae'],\
					 '%.2f' % f_err['rmse']])

# Sort according to sigma.
data = sorted(data, key=lambda data_col: data_col[4]) 

print ' '*74 + 'Energy' + ' '*14 + 'Forces'
print ' '*74 + '-'*13 + ' '*7 + '-'*13
print ('{:<' + str(name_str_maxlen) + '}  ' + '{:<' + str(theory_str_maxlen) + '}  ' + '{:>8}  '*3 + '{:>8}  '*4).format('', *data_names)
print '-'*107

row_format ='{:<' + str(name_str_maxlen) + '}  ' + '{:<' + str(theory_str_maxlen) + '}  ' + '{:>8}  '*3 + '{:>8}  '*4
for row in data:
	print row_format.format(*row) + ('*' if row[4] == best_sig else '')
print ''

if not os.path.exists(best_model_dir):
	os.makedirs(best_model_dir)

if os.path.isfile(best_model_dir + best_path) and args.overwrite:	
	print ui.info_str('[INFO]') + ' Overwriting existing model file in \'models/assist/' + best_path + '\'.'

if not os.path.isfile(best_model_dir + best_path) or args.overwrite:
	print 'Writing model file to \'models/assist/' + best_path + '\'...'
	shutil.copy(args.model_dir + '/' + best_path, best_model_dir + best_path)
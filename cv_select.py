#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import shutil

import numpy as np

from src.utils import ui

best_model_dir = BASE_DIR + '/models/auto/'


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
data_names = ['Theory Level', 'N train', 'N test', 'sig', 'MAE', 'RMSE', 'MAE', 'RMSE']
dataset_str_maxlen = 0
theory_lavel_str_maxlen = len(data_names[1])
for file in sorted(os.listdir(args.model_dir)): # iterate in (lexicographic) order
	if file.endswith(".npz"):
		try:
			model = np.load(args.model_dir + '/' + file)
		except:
			sys.exit("ERROR: Reading model file failed.")

		theory_level_str = '?'
		if 'theory_level' in model:
			theory_level_str = str(model['theory_level'])

		n_test = np.nan
		if 'n_test' in model:
			n_test = model['n_test']

		if model['f_rmse'] < best_sig_f_rmse:
			best_sig_f_rmse = model['f_rmse']
			best_sig = model['sig']
			best_path = file

		dataset_str_maxlen = max(dataset_str_maxlen,len(str(model['dataset'])))
		theory_lavel_str_maxlen = max(theory_lavel_str_maxlen,len(theory_level_str))
		data.append([model['dataset'],\
					 theory_level_str,\
					 str(model['Rt_desc'].shape[1]),\
					 n_test,\
					 model['sig'],\
					 "%.2f" % (model['e_mae']),\
					 "%.2f" % (model['e_rmse']),\
					 "%.2f" % (model['f_mae']),\
					 "%.2f" % (model['f_rmse'])])

# Sort according to sigma.
data = sorted(data, key=lambda data_col: data_col[4]) 

print ' '*74 + 'Energy' + ' '*14 + 'Forces'
print ' '*74 + '-'*13 + ' '*7 + '-'*13
print ('{:<' + str(dataset_str_maxlen) + '}  ' + '{:<' + str(theory_lavel_str_maxlen) + '}  ' + '{:>8}  '*3 + '{:>8}  '*4).format('', *data_names)
print '-'*107

row_format ='{:<' + str(dataset_str_maxlen) + '}  ' + '{:<' + str(theory_lavel_str_maxlen) + '}  ' + '{:>8}  '*3 + '{:>8}  '*4
for row in data:
	print row_format.format(*row) + ('*' if row[4] == best_sig else '')
print ''

if not os.path.exists(best_model_dir):
	os.makedirs(best_model_dir)

if os.path.isfile(best_model_dir + best_path) and args.overwrite:	
	print ui.info_str('[INFO]') + ' Overwriting existing model file in \'models/auto/' + best_path + '\'.'

if not os.path.isfile(best_model_dir + best_path) or args.overwrite:
	print 'Writing model file to \'models/auto/' + best_path + '\'...'
	shutil.copy(args.model_dir + '/' + best_path, best_model_dir + best_path)
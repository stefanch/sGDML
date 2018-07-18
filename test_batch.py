#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse

import numpy as np

from src.utils import ui


parser = argparse.ArgumentParser(description='Tests all models in given directory.')
parser.add_argument('model_dir', metavar = '<model_dir>',\
							 	 type    = lambda x: ui.is_dir(x),\
							 	 help	 = 'path to model directory')
parser.add_argument('dataset',   metavar = '<dataset>',\
							     type    = lambda x: ui.is_valid_np_file(parser, x),\
							     help	= 'path to dataset file')
parser.add_argument('n_test', 	 metavar = '<n_test>',\
							  	 type    = lambda x: ui.is_strict_pos_int(x),\
							  	 help    = 'number of test points from dataset',\
							  	 nargs   = '?', default = None)
parser.add_argument('-s', '--silent', dest='silent', action='store_true', help = 'suppress output')
parser.add_argument('-u', '--update', dest='update', action='store_true', help = 'update expected prediction errors in model file')
args = parser.parse_args()
dataset_path, _ = args.dataset


model_files = []
for model_file in sorted(os.listdir(args.model_dir)):
	if model_file.endswith('.npz'):
		model_path = args.model_dir + '/' + model_file
		try:
			model = np.load(model_path)
		except:
			sys.exit("ERROR: Reading file failed.")
		if 'type' in model and model['type'] == 'm':
			model_files.append(model_file)

#model_files = [model_file for model_file in sorted(os.listdir(args.model_dir)) if model_file.endswith('.npz')]
if not len(model_files):
	sys.exit(os.path.basename(sys.argv[0]) + ': error: no models found in directory')

for i,model_file in enumerate(model_files):
	print 'Testing model %d/%d...' % (i+1,len(model_files))
	os.system('python test.py ' + args.model_dir + '/' + model_file + ' ' + dataset_path\
		   + (' ' + str(args.n_test) if args.n_test else '')\
		   + (' -s' if args.silent else '')\
		   + (' -u' if args.update else ''))
	print ''

if not args.silent:
	call_str = 'python cv_select.py ' + args.model_dir
	print '\nNEXT STEP: \'' + call_str + '\''
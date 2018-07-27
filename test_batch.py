#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse

import numpy as np

from src.utils import ui


parser = argparse.ArgumentParser(description='Tests all models in given directory.')
parser.add_argument('model_dir', metavar = '<model_dir>',\
							 	 type    = lambda x: ui.is_dir_with_file_type(x, 'model'),\
							 	 help	 = 'path to model directory')
parser.add_argument('dataset',   metavar = '<dataset>',\
							     type    = lambda x: ui.is_file_type(x, 'dataset'),\
							     help	= 'path to dataset file')
parser.add_argument('n_test', 	 metavar = '<n_test>',\
							  	 type    = lambda x: ui.is_strict_pos_int(x),\
							  	 help    = 'number of test points from dataset',\
							  	 nargs   = '?', default = None)
parser.add_argument('-s', '--silent', dest='silent', action='store_true', help = 'suppress output')
parser.add_argument('-u', '--update', dest='update', action='store_true', help = 'update expected prediction errors in model file')
args = parser.parse_args()
dataset_path, _ = args.dataset

model_files, model_dir = args.model_dir


for i,model_file in enumerate(model_files):
	print 'Testing model %d/%d...' % (i+1,len(model_files))
	os.system('python test.py ' + model_dir + '/' + model_file + ' ' + dataset_path\
		   + (' ' + str(args.n_test) if args.n_test else '')\
		   + (' -s' if args.silent else '')\
		   + (' -u' if args.update else ''))
	print ''

if not args.silent:
	call_str = 'python cv_select.py ' + model_dir
	print '\nNEXT STEP: \'' + call_str + '\''
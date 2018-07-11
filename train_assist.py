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

from src.utils import ui

fail_msg = ui.fail_str('[FAIL]') + ' Training assistant failed.'


parser = argparse.ArgumentParser(description='Performs all steps necessary to train a sGDML model from a given dataset.')
parser.add_argument('dataset', metavar = '<dataset>',\
							   type    = lambda x: ui.is_valid_np_file(parser, x),\
							   help	   = 'path to dataset file')
parser.add_argument('n_train', metavar = '<n_train>',\
							   type    = lambda x: ui.is_strict_pos_int(x),\
							   help    = 'number of training points to sample from dataset')
parser.add_argument('n_test',  metavar = '<n_test>',\
							   type    = lambda x: ui.is_strict_pos_int(x),\
							   help    = 'number of test points form dataset to validate each model',\
							   nargs   = '?', default = None)
parser.add_argument('--gdml', dest='gdml', action='store_true', help = 'don\'t include symmetries in the model')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing results')
args = parser.parse_args()
dataset_path, dataset = args.dataset


print ''
print '-'*100
print 'sGDML Model Creation Assistant'
print '-'*100

print '\nSTEP 1: Create cross-validation tasks.'
print '-'*100
err_code = os.system('python cv_schedule.py ' + dataset_path + ' ' + str(args.n_train)\
				  + (' --gdml' if args.gdml else '')\
				  + (' -o' if args.overwrite else '')\
				  +  ' -s')
if err_code != 0: sys.exit(fail_msg)


print '\nSTEP 2: Train all models.'
print '-'*100
theory_level_str = re.sub('[^\w\-_\.]', '_', str(dataset['theory_level']))
theory_level_str = re.sub('__', '_', theory_level_str)
dataset_name_str = str(dataset['name'])
task_dir = 'training/' + dataset_name_str + '-' + theory_level_str + '-' + str(args.n_train)
err_code = os.system('python train_batch.py ' + task_dir\
				  + (' -o' if args.overwrite else '')\
				  +  ' -s')
if err_code != 0: sys.exit(fail_msg)


print '\nSTEP 3: Test all models.'
print '-'*100
err_code = os.system('python test_batch.py ' + task_dir + ' ' + dataset_path + ' ' + (str(args.n_test) if args.n_test else '')\
				   + ' -s')
if err_code != 0: sys.exit(fail_msg)


print '\nSTEP 4: Select best hyper-parameter combination.'
print '-'*100
err_code = os.system('python cv_select.py ' + task_dir\
				  + (' -o' if args.overwrite else ''))
if err_code != 0: sys.exit(fail_msg)

print ui.pass_str('[DONE]') + ' Training assistant finished sucessfully.'
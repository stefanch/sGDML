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



#print ui.warn_str('[WARN]') + ' Geometries, forces and energies must have consistent units.'


print ''
print '-'*100
print 'sGDML Model Creation Assistant'
print '-'*100

print '\n' + ui.white_back_str(' STEP 1 ') + ' Splitting dataset.'
print '-'*100
err_code = os.system('python split_dataset.py ' + dataset_path + ' ' + str(args.n_train) + ' ' + str(args.n_test)\
 				  + (' ' + str(args.n_valid) if args.n_valid else '')\
				  + (' --gdml' if args.gdml else '')\
				  + (' -o' if args.overwrite else ''))
if err_code != 0: sys.exit(fail_msg)

dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
train_dataset_path = 'datasets/npz/splits/%s/train_%s.npz' % (dataset_name, dataset_name)
test_dataset_path = 'datasets/npz/splits/%s/test_%s.npz' % (dataset_name, dataset_name)
valid_dataset_path = 'datasets/npz/splits/%s/valid_%s.npz' % (dataset_name, dataset_name)


print '\n' + ui.white_back_str(' STEP 2 ') + ' Creating cross-validation tasks.'
print '-'*100
err_code = os.system('python cv_schedule.py ' + train_dataset_path + ' ' + str(args.n_train)\
				  + (' --gdml' if args.gdml else '')\
				  + (' -o' if args.overwrite else '')\
				  +  ' -s')
if err_code != 0: sys.exit(fail_msg)


print '\n' + ui.white_back_str(' STEP 3 ') + ' Training all models.'
print '-'*100
theory_level_str = re.sub('[^\w\-_\.]', '_', str(dataset['theory_level']))
theory_level_str = re.sub('__', '_', theory_level_str)
dataset_name_str = str(dataset['name'])
task_dir = 'training/' + dataset_name_str + '-' + theory_level_str + '-' + str(args.n_train)
err_code = os.system('python train_batch.py ' + task_dir\
				  + (' -o' if args.overwrite else '')\
				  +  ' -s')
if err_code != 0: sys.exit(fail_msg)


print '\n' + ui.white_back_str(' STEP 4 ') + ' Testing all models.'
print '-'*100
err_code = os.system('python test_batch.py ' + task_dir + ' ' + test_dataset_path + ' ' + str(args.n_test)\
				   + ' -s')
if err_code != 0: sys.exit(fail_msg)


print '\n' + ui.white_back_str(' STEP 5 ') + ' Select best hyper-parameter combination.'
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
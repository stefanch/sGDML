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

from src.gdml_train import GDMLTrain
from src.utils import io, ui


parser = argparse.ArgumentParser(description='Creates sGDML model training tasks for a specified hyper-parameter range.')
parser.add_argument('dataset', metavar = '<dataset>',\
							   type    = lambda x: ui.is_valid_np_file(parser, x),\
							   help	   = 'path to dataset file')
parser.add_argument('n_train', metavar = '<n_train>',\
							   type    = lambda x: ui.is_strict_pos_int(x),\
							   help    = 'number of data points to use for training')
parser.add_argument('--gdml', dest='gdml', action='store_true', help = 'don\'t include symmetries in the model')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing training directory')
parser.add_argument('-s', '--silent', dest='silent', action='store_true', help = 'suppress output')
args = parser.parse_args()
_, dataset = args.dataset

gdml = GDMLTrain()

lam = 1e-15
task = gdml.create_task(dataset, args.n_train, 1, lam, not args.gdml)

theory_level_str = re.sub('[^\w\-_\.]', '_', str(dataset['theory_level']))
theory_level_str = re.sub('__', '_', theory_level_str)
dataset_name_str = str(dataset['name'])
task_dir = BASE_DIR + '/training/' + dataset_name_str + '-' + theory_level_str + '-' + str(args.n_train)

if os.path.exists(task_dir):
	if args.overwrite:
		print '[INFO] Overwriting existing training directory.'
		file_list = [f for f in os.listdir(task_dir)]
		for f in file_list:
			os.remove(os.path.join(task_dir, f))
else:
	os.makedirs(task_dir)

print 'Writing tasks with %s training points each.' % task['R'].shape[0]
for sig in range(2,100,4):
	task['sig'] = sig

	task_path = task_dir + '/task-' + io.task_file_name(task)
	if os.path.isfile(task_path + '.mat'):
		print '[INFO] Skipping exising task \'task-' + io.task_file_name(task) + '.mat\'.'
	else:
		try:
			scipy.io.savemat(task_path, task)
		except:
			sys.exit('  ERROR: Writing train task (\'' + task_dir + '\') failed.')

if not args.silent:
	call_str = 'python train_batch.py training/' + dataset_name_str + '-' + theory_level_str + '-' + str(args.n_train)
	print '\nNEXT STEP: \'' + call_str + '\''
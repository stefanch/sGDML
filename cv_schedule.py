#!/usr/bin/python

# GDML Force Field
# Author: Stefan Chmiela (stefan@chmiela.com)

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import re

import numpy as np

from src.gdml_train import GDMLTrain
from src.utils import io, ui


parser = argparse.ArgumentParser(description='Creates sGDML model training tasks for a specified hyper-parameter range.')
parser.add_argument('dataset', metavar = '<[train_]dataset>',\
							   type    = lambda x: ui.is_file_type(x, 'dataset'),\
							   help	   = 'path to dataset file (train and test data are both sampled from here, if no separate test set is specified)')
parser.add_argument('n_train', metavar = '<n_train>',\
							   type    = lambda x: ui.is_strict_pos_int(x),\
							   help    = 'number of data points to use for training')
parser.add_argument('n_test', metavar = '<n_test>',\
							   type    = lambda x: ui.is_strict_pos_int(x),\
							   help    = 'number of data points to use for testing')
parser.add_argument('--test_dataset', metavar = '<test_dataset>', nargs='?',\
							   type    = lambda x: ui.is_file_type(x, 'dataset'),\
							   help	   = 'path to test dataset file')
parser.add_argument('--gdml', dest='gdml', action='store_true', help = 'don\'t include symmetries in the model')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing training directory')
parser.add_argument('-s', '--silent', dest='silent', action='store_true', help = 'suppress output')
args = parser.parse_args()
_, dataset = args.dataset

if args.test_dataset is not None:
	test_dataset_path, test_dataset = args.test_dataset
else:
	test_dataset_path = dataset_path
	test_dataset = dataset


gdml = GDMLTrain()

if dataset['E'].shape[0] < args.n_train:
	sys.exit(ui.fail_str('[FAIL]') + ' Dataset only contains {} points, can not train on {} points.'.format(dataset['E'].shape[0],args.n_train)) 
elif args.test_dataset is None and dataset['E'].shape[0]-args.n_train < args.n_test:
	sys.exit(ui.fail_str('[FAIL]') + ' Dataset only contains {} points, can not train on {} and test on {} points.'.format(dataset['E'].shape[0],args.n_train,args.n_test)) 
elif args.test_dataset is not None and test_dataset['E'].shape[0] < args.n_test:
	sys.exit(ui.fail_str('[FAIL]') + ' Test dataset only contains {} points, can not test on {} points.'.format(test_dataset['E'].shape[0],args.n_test)) 

theory_level_str = re.sub('[^\w\-_\.]', '_', str(dataset['theory']))
theory_level_str = re.sub('__', '_', theory_level_str)
dataset_name_str = str(dataset['name'])
task_dir = BASE_DIR + '/training/' + dataset_name_str + '-' + theory_level_str + '-' + str(args.n_train)

if os.path.exists(task_dir):
	if args.overwrite:
		print ' ' + ui.info_str('[INFO]') + ' Overwriting existing training directory.'
		file_list = [f for f in os.listdir(task_dir)]
		for f in file_list:
			os.remove(os.path.join(task_dir, f))
	else:
		print ui.warn_str('[WARN]') + ' Skipping existing task dir \'%s\'.' % task_dir
		sys.exit()
else:
	os.makedirs(task_dir)

lam = 1e-15
task = gdml.create_task(dataset, args.n_train, test_dataset, args.n_test, sig=1, lam=lam, recov_sym=not args.gdml)

print 'Writing tasks with %s training points each.' % task['R_train'].shape[0]
for sig in range(2,100,4):
	task['sig'] = sig

	task_path = task_dir + '/task-' + io.task_file_name(task)
	if os.path.isfile(task_path + '.npz'):
		print ' ' + ui.info_str('[INFO]') + ' Skipping exising task \'task-' + io.task_file_name(task) + '.npz\'.'
	else:
		try:
			np.savez_compressed(task_path, **task)
		except:
			sys.exit('  ERROR: Writing train task (\'' + task_dir + '\') failed.')
print ''

if not args.silent:
	call_str = 'python train_batch.py training/' + dataset_name_str + '-' + theory_level_str + '-' + str(args.n_train)
	print '\nNEXT STEP: \'' + call_str + '\''
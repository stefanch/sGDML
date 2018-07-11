#!/usr/bin/python

# GDML Force Field
# Author: Stefan Chmiela (stefan@chmiela.com)

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import random
import string


import numpy as np

from src.gdml_train import GDMLTrain
from src.utils import io,ui


parser = argparse.ArgumentParser(description='Trains the sGDML model for a given task.')
parser.add_argument('task', metavar = '<task>',\
							type    = lambda x: ui.is_valid_mat_file(parser, x),\
							help	= 'path to task file')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing model')
parser.add_argument('-s', '--silent', dest='silent', action='store_true', help = 'suppress output')
args = parser.parse_args()
task_path, task = args.task


model_path = os.path.dirname(task_path) + '/model-' + io.task_file_name(task)
if os.path.isfile(model_path) and not args.overwrite:
	sys.exit('[INFO] Skipping exising model \'model-' + io.task_file_name(task) + '\'.')

gdml = GDMLTrain()
model = gdml.train(task)

if not args.silent: print "Recovering integration constant..."
model['c'] = gdml.recov_int_const(model)

if not args.silent: print 'Writing model to file \'' + model_path + '\'...'
np.savez_compressed(model_path, **model)

if not args.silent:
	call_str = 'python test.py ' + model_path + ' <dataset> <n_test>'
	print '\nNEXT STEP: \'' + call_str + '\''
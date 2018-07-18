#!/usr/bin/python

# GDML Force Field
# Author: Stefan Chmiela (stefan@chmiela.com)

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import time

import numpy as np

import multiprocessing as mp

from src.gdml_predict import GDMLPredict
from src.utils import ui

parser = argparse.ArgumentParser(description='Benchmarks a given sGDML model to determine optimal paralell processing parameters.')
parser.add_argument('model', metavar = '<model>',\
							 type    = lambda x: ui.is_valid_np_file(parser, x),\
							 help	 = 'path to model file')
parser.add_argument('n_reps', metavar = '<n_reps>',\
							  type    = lambda x: ui.is_strict_pos_int(x),\
							  help    = 'number of repetitions per test',\
							  nargs   = '?', default = 100)
parser.add_argument('-s', '--silent', dest='silent', action='store_true', help = 'suppress output')
args = parser.parse_args()
model_path, model = args.model
n_reps = args.n_reps


n_train = model['R_desc'].shape[1]
n_train = (n_train if n_train % 10 == 0 else n_train + 10 - n_train % 10) # round to next 10
n_atoms = model['z'].shape[0]

if not args.silent:
	print
	print '------------------- Benchmark -------------------------'
	print "Dataset: '%s'%s" % (str(model['dataset_name']), str(model['dataset_theory']))
	print 'Sequential evaluation test (one geometry at a time)'
	print '-------------------------------------------------------'

best_sps = 0
best_num_workers = None
best_batch_size = None

r_dummy = np.random.rand(1,n_atoms*3)
for num_workers in [1] + range(2,mp.cpu_count()+2,2):
	if n_train % num_workers == 0:

		if not args.silent:
			print '\n' + str(num_workers) + ' Worker(s)'
			print '-' * 25
			print 'Batch Size | Performance '
			print '-' * 25

		max_wkr_size = int(np.ceil(n_train / num_workers))

		#for batch_size in range(10,n_train,10):
		for batch_size in range(1,max_wkr_size/2+1) + [max_wkr_size]:
			if max_wkr_size % batch_size == 0:
				gdml = GDMLPredict(model, batch_size, num_workers)

				t = time.time()
				for i in range(n_reps):
					gdml.predict(r_dummy)

				sps = n_reps / (time.time() - t)
				if sps > best_sps:
					best_sps = sps
					best_num_workers = num_workers
					best_batch_size = batch_size

				if not args.silent: print '{:10d} | {:7.2f} sps --- {:10d}'.format(batch_size, sps, batch_size*n_atoms*model['perms'].shape[0]*n_atoms)

print '\nBest performance: {:3.2f} sps'.format(best_sps)
print '| num_worker = {:3d}'.format(best_num_workers)
print '| batch_size = {:3d}'.format(best_batch_size)
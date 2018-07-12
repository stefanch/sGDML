#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse

from src.utils import ui


parser = argparse.ArgumentParser(description='Trains all tasks in given directory.')
parser.add_argument('task_dir', metavar = '<task_dir>',\
							 	type    = lambda x: ui.is_dir(x),\
							 	help	 = 'path to task directory')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing model')
parser.add_argument('-s', '--silent', dest='silent', action='store_true', help = 'suppress output')
args = parser.parse_args()


task_files = [task_file for task_file in sorted(os.listdir(args.task_dir)) if task_file.endswith('.mat')]
if not len(task_files):
	sys.exit(os.path.basename(sys.argv[0]) + ': error: no tasks found in directory')

for i,task_file in enumerate(task_files):
	print 'Training task %d/%d...' % (i+1,len(task_files))
	os.system('python train.py ' + args.task_dir + '/' + task_file\
		   + (' -o' if args.overwrite else '')\
		   + (' -s' if args.silent else ''))
	print ''

if not args.silent:
	call_str = 'python test_batch.py ' + args.task_dir + ' <dataset>' + ' <n_test>'
	print '\nNEXT STEP: \'' + call_str + '\''
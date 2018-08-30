#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse
import numpy as np

from src.utils import io,ui

dataset_dir = BASE_DIR + '/datasets/xyz/'


parser = argparse.ArgumentParser(description='Creates a dataset from extended XYZ format.')
parser.add_argument('dataset', metavar = '<dataset>',\
							 type    = lambda x: ui.is_file_type(x, 'dataset'),\
							 help	 = 'path to dataset file')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing xyz dataset file')

args = parser.parse_args()
dataset_path, dataset = args.dataset


if not os.path.exists(dataset_dir):
	os.makedirs(dataset_dir)

name = os.path.splitext(os.path.basename(dataset_path))[0]
dataset_path = dataset_dir + name + '.xyz'


xyz_exists = os.path.isfile(dataset_path)
if xyz_exists and args.overwrite:	
	print ui.info_str('[INFO]') + ' Overwriting existing xyz dataset file.'
if not xyz_exists or args.overwrite:
	print 'Writing dataset to \'datasets/xyz/%s.xyz\'...' % name
else:
	sys.exit(ui.fail_str('[FAIL]') + ' Dataset \'datasets/xyz/%s.xyz\' already exists.' % name)

z = dataset['z']
R = dataset['R']
F = dataset['F']
E = dataset['E']
n = R.shape[0]

try:
	with open(dataset_path,'w') as f:
		for i,r in enumerate(R):

			f.write(str(len(r)) + '\n' + str(np.squeeze(E[i])))
			for j,atom in enumerate(r):
				f.write('\n' + io._z_to_z_str_dict[z[j]] + '\t')
				f.write('\t'.join(str(x) for x in atom) + '\t')
				f.write('\t'.join(str(x) for x in F[i][j]))
			f.write('\n')

			progr = float(i) / (n-1)
			sys.stdout.write('\r\x1b[1;37m[%3d%%]\x1b[0m Exporting %d data points...' % ((progr * 100),n))
			sys.stdout.flush()
except IOError:
	sys.exit("ERROR: Writing xyz file failed.")

print '\n' + ui.pass_str('DONE')
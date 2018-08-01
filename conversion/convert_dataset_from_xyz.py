#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse
import numpy as np

from src.utils import io,ui

dataset_dir = BASE_DIR + '/datasets/npz/'

# Assumes that the atoms in each molecule are in the same order.
def read_concat_ext_xyz(f):
	n_atoms = None

	R,z,E,F = [],[],[],[]
	for i,line in enumerate(f):
		line = line.strip()
		if not n_atoms:
			n_atoms = int(line)
			print '| Number atoms per geometry:      {:>7d}'.format(n_atoms)

		file_i, line_i = divmod(i, n_atoms+2)

		if line_i == 1:
			E.append(float(line))

		cols = line.split()
		if line_i >= 2:
			R.append(map(float,cols[1:4]))
			if file_i == 0: # first molecule
				z.append(io._z_str_to_z_dict[cols[0]])
			F.append(map(float,cols[4:7]))

		if file_i % 1000 == 0:
			sys.stdout.write("\r| Number geometries found so far: {:>7d}".format(file_i))
			sys.stdout.flush()
	sys.stdout.write("\r| Number geometries found so far: {:>7d}".format(file_i))
	sys.stdout.flush()
	print ''

	R = np.array(R).reshape(-1,n_atoms,3)
	z = np.array(z)
	E = np.array(E)
	F = np.array(F).reshape(-1,n_atoms,3)

	f.close()
	return (R,z,E,F)


parser = argparse.ArgumentParser(description='Creates a dataset from extended XYZ format.')
parser.add_argument('dataset', metavar = '<dataset>',\
							   type    = argparse.FileType('r'),\
							   help	   = 'path to xyz dataset file')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing dataset file')
args = parser.parse_args()
dataset = args.dataset


if not os.path.exists(dataset_dir):
	os.makedirs(dataset_dir)
name = os.path.splitext(os.path.basename(dataset.name))[0]
dataset_path = dataset_dir + name + '.npz'

dataset_exists = os.path.isfile(dataset_path)
if dataset_exists and args.overwrite:	
	print ui.info_str('[INFO]') + ' Overwriting existing dataset file.'
if not dataset_exists or args.overwrite:
	print 'Writing dataset to \'datasets/npz/%s.npz\'...' % name
else:
	sys.exit(ui.fail_str('[FAIL]') + ' Dataset \'datasets/npz/%s.npz\' already exists.' % name)


R,z,E,F = read_concat_ext_xyz(dataset)

# Base variables contained in every model file.
base_vars = {'type':			'd',\
			 'R':				R,\
			 'z':				z,\
			 'E':				E[:,None],\
			 'F':				F,\
			 'name':			name,\
			 'theory':			'unknown'}
base_vars['md5'] = io.dataset_md5(base_vars)

np.savez_compressed(dataset_path, **base_vars)
print ui.pass_str('DONE')
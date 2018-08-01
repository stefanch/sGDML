#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse
import numpy as np

from src.utils import io,ui

dataset_dir = BASE_DIR + '/datasets/npz/'


def read_reference_data(f):
	eV_to_kcalmol = 0.036749326/0.0015946679

	e_next, f_next, geo_next = False, False, False
	n_atoms = None
	R,z,E,F = [],[],[],[]

	
	geo_idx = 0
	for line in f:
		if n_atoms:
			cols = line.split()
			if e_next:
				E.append(float(cols[5]))
				e_next = False
			elif f_next:
				a = int(cols[1])-1
				F.append(map(float,cols[2:5]))
				if a == n_atoms-1:
					f_next = False
			elif geo_next:
				if 'atom' in cols:
					a_count += 1
					R.append(map(float,cols[1:4]))

					if geo_idx == 0:
						z.append(io._z_str_to_z_dict[cols[4]])

					if a_count == n_atoms:
						geo_next = False
						geo_idx += 1
			elif 'Energy and forces in a compact form:' in line:
					e_next = True
			elif 'Total atomic forces (unitary forces cleaned) [eV/Ang]:' in line:
					f_next = True
			elif 'Atomic structure (and velocities) as used in the preceding time step:' in line:
					geo_next = True
					a_count = 0
		elif 'The structure contains' in line and 'atoms,  and a total of' in line:
			n_atoms = int(line.split()[3])
			print '| Number atoms per geometry:      {:>7d}'.format(n_atoms)
			continue

		if geo_idx > 0 and geo_idx % 1000 == 0:
			sys.stdout.write("\r| Number geometries found so far: {:>7d}".format(geo_idx))
			sys.stdout.flush()
	sys.stdout.write("\r| Number geometries found so far: {:>7d}".format(geo_idx))
	sys.stdout.flush()
	print '\n' + ui.info_str('[INFO]') + ' Energies and forces have been converted from eV to kcal/mol(/Ang)'


	R = np.array(R).reshape(-1,n_atoms,3) 
	z = np.array(z)
	E = np.array(E) * eV_to_kcalmol
	F = np.array(F).reshape(-1,n_atoms,3) * eV_to_kcalmol

	f.close()
	return (R,z,E,F)


parser = argparse.ArgumentParser(description='Creates a dataset from FHI-aims format.')
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


R,z,E,F = read_reference_data(dataset)

# Prune all arrays to same length.
n_mols = min(min(R.shape[0], F.shape[0]), E.shape[0])
if n_mols != R.shape[0] or n_mols != F.shape[0] or n_mols != E.shape[0]:
	print ui.warn_str('[WARN]') + ' Incomplete output detected: Final dataset was pruned to %d points.' % n_mols
R = R[:n_mols,:,:]
F = F[:n_mols,:,:]
E = E[:n_mols]

# Base variables contained in every model file.
base_vars = {'type':			'd',\
			 'R':				R,\
			 'z':				z,\
			 'E':				E[:,None],\
			 'F':				F,\
			 'name':			name,\
			 'theory':	'unknown'}

np.savez_compressed(dataset_path, **base_vars)
print ui.pass_str('DONE')
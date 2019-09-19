#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018 Stefan Chmiela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import argparse
import os
import sys

from ase.io import read
import numpy as np

from sgdml.utils import io, ui


parser = argparse.ArgumentParser(
    description='Creates a dataset from extended XYZ format.'
)
parser.add_argument(
    'dataset',
    metavar='<dataset>',
    type=argparse.FileType('r'),
    help='path to extended xyz dataset file',
)
parser.add_argument(
    '-o',
    '--overwrite',
    dest='overwrite',
    action='store_true',
    help='overwrite existing dataset file',
)
args = parser.parse_args()
dataset = args.dataset


name = os.path.splitext(os.path.basename(dataset.name))[0]
dataset_file_name = name + '.npz'

dataset_exists = os.path.isfile(dataset_file_name)
if dataset_exists and args.overwrite:
    print(ui.info_str('[INFO]') + ' Overwriting existing dataset file.')
if not dataset_exists or args.overwrite:
    print('Writing dataset to \'%s\'...' % dataset_file_name)
else:
    sys.exit(
        ui.fail_str('[FAIL]') + ' Dataset \'%s\' already exists.' % dataset_file_name
    )

mols = read(dataset, index=':')
print("\rNumber geometries found: {:>7d}\n".format(len(mols)))

#cell = np.array(mols[0].get_cell())
#print(cell)

Z = np.array([mol.get_atomic_numbers() for mol in mols])
all_z_the_same = (Z == Z[0]).all()
if not all_z_the_same:
  sys.exit(
    ui.fail_str('[FAIL]') + ' Order of atoms changes within dataset.'
  )

# Base variables contained in every model file.
base_vars = {
   'type': 'd',
   'lattice': np.array(mols[0].get_cell()),
   'R': np.array([mol.get_positions() for mol in mols]),
   'z': Z[0],
   'E': np.array([mol.get_potential_energy() for mol in mols])[:, None],
   'F': np.array([mol.get_forces() for mol in mols]),
   'name': name,
   'theory': 'unknown',
}
base_vars['md5'] = io.dataset_md5(base_vars)

np.savez_compressed(dataset_file_name, **base_vars)
print(ui.pass_str('DONE'))

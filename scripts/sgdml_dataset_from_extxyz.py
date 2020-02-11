#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2019 Stefan Chmiela
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

#import ase.units
from ase.io import read
import numpy as np

from sgdml import __version__
from sgdml.utils import io, ui

if sys.version[0] == '3':
    raw_input = input


# Assumes that the atoms in each molecule are in the same order.
def read_nonstd_ext_xyz(f):
    n_atoms = None

    R, z, E, F = [], [], [], []
    for i, line in enumerate(f):
        line = line.strip()
        if not n_atoms:
            n_atoms = int(line)
            print('Number atoms per geometry: {:,}'.format(n_atoms))

        file_i, line_i = divmod(i, n_atoms + 2)

        if line_i == 1:
            try:
                e = float(line)
            except ValueError:
                pass
            else:
                E.append(e)

        cols = line.split()
        if line_i >= 2:
            R.append(list(map(float, cols[1:4])))
            if file_i == 0:  # first molecule
                z.append(io._z_str_to_z_dict[cols[0]])
            F.append(list(map(float, cols[4:7])))

        if file_i % 1000 == 0:
            sys.stdout.write('\rNumber geometries found so far: {:,}'.format(file_i))
            sys.stdout.flush()
    sys.stdout.write('\rNumber geometries found so far: {:,}'.format(file_i))
    sys.stdout.flush()
    print()

    R = np.array(R).reshape(-1, n_atoms, 3)
    z = np.array(z)
    E = None if not E else np.array(E)
    F = np.array(F).reshape(-1, n_atoms, 3)

    if F.shape[0] != R.shape[0]:
        sys.exit(
            ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
            + ' Force labels are missing from dataset or are incomplete!'
        )

    f.close()
    return (R, z, E, F)


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
    print(ui.color_str('[INFO]', bold=True) + ' Overwriting existing dataset file.')
if not dataset_exists or args.overwrite:
    print('Writing dataset to \'{}\'...'.format(dataset_file_name))
else:
    sys.exit(
        ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
        + ' Dataset \'{}\' already exists.'.format(dataset_file_name)
    )

mols = read(dataset.name, index=':')

lattice, R, z, E, F = None, None, None, None, None

calc = mols[0].get_calculator()
is_extxyz = calc is not None
if is_extxyz:

    print("\rNumber geometries found: {:,}\n".format(len(mols)))

    if 'forces' not in calc.results:
        sys.exit(
            ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
            + ' Forces are missing in the input file!'
        )

    lattice = np.array(mols[0].get_cell())
    if not np.any(lattice):
        print(
            ui.color_str('[INFO]', bold=True)
            + ' No lattice vectors specified in extended XYZ file.'
        )

    Z = np.array([mol.get_atomic_numbers() for mol in mols])
    all_z_the_same = (Z == Z[0]).all()
    if not all_z_the_same:
        sys.exit(
            ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
            + ' Order of atoms changes accross dataset.'
        )

    lattice = np.array(mols[0].get_cell())
    if not np.any(lattice): # all zeros
        lattice = None

    R = np.array([mol.get_positions() for mol in mols])
    z = Z[0]

    if 'Energy' in mols[0].info:
        E = np.array([mol.info['Energy'] for mol in mols])
    if 'energy' in mols[0].info:
        E = np.array([mol.info['energy'] for mol in mols])
    F = np.array([mol.get_forces() for mol in mols])

else:  # legacy non-standard XYZ format

    with open(dataset.name) as f:
        R, z, E, F = read_nonstd_ext_xyz(f)


# Base variables contained in every model file.
base_vars = {
    'type': 'd',
    'code_version': __version__,
    'name': name,
    'theory': 'unknown',
    'R': R,
    'z': z,
    'F': F,
}

base_vars['F_min'], base_vars['F_max'] = np.min(F.ravel()), np.max(F.ravel())
base_vars['F_mean'], base_vars['F_var'] = np.mean(F.ravel()), np.var(F.ravel())

print('Please provide a description of the length unit used in your input file, e.g. \'Ang\' or \'au\': ')
print('Note: This string will be stored in the dataset file and passed on to models files for later reference.')
r_unit = raw_input('> ').strip()
if r_unit != '':
    base_vars['r_unit'] = r_unit

print('Please provide a description of the energy unit used in your input file, e.g. \'kcal/mol\' or \'eV\': ')
print('Note: This string will be stored in the dataset file and passed on to models files for later reference.')
e_unit = raw_input('> ').strip()
if e_unit != '':
    base_vars['e_unit'] = e_unit

if E is not None:
    base_vars['E'] = E
    base_vars['E_min'], base_vars['E_max'] = np.min(E), np.max(E)
    base_vars['E_mean'], base_vars['E_var'] = np.mean(E), np.var(E)
else:
    print(ui.color_str('[INFO]', bold=True) + ' No energy labels found in dataset.')

if lattice is not None:
    base_vars['lattice'] = lattice

base_vars['md5'] = io.dataset_md5(base_vars)
np.savez_compressed(dataset_file_name, **base_vars)
print(ui.color_str('[DONE]', fore_color=ui.GREEN, bold=True))

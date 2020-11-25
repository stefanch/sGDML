#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2020 Stefan Chmiela
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

try:
    from ase.io import read
except ImportError:
    raise ImportError('Optional ASE dependency not found! Please run \'pip install sgdml[ase]\' to install it.')

import numpy as np

from sgdml import __version__
from sgdml.utils import io, ui

if sys.version[0] == '3':
    raw_input = input


parser = argparse.ArgumentParser(
    description='Creates a dataset from any input format supported by ASE.'
)
parser.add_argument(
    'dataset',
    metavar='<dataset>',
    type=argparse.FileType('r'),
    help='path to input dataset file',
    nargs='+',
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

if len(dataset) == 1:
    name = os.path.splitext(os.path.basename(dataset.name))[0]
else:
    import difflib

    def name_overlap(name1, name2):
        i, _, lenght = difflib.SequenceMatcher(None, name1, name2).find_longest_match(0, len(name1), 0, len(name2))
        return name1[i:i + lenght]

    base_name = []
    for idx1, name1 in enumerate(dataset):
        for name2 in dataset[idx1+1:]:
            base_name.append(name_overlap(name1.name, name2.name))

    name = list(set(base_name))[0]
    name = os.path.splitext(os.path.basename(name))[0]

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

lattice_new, R_new, z_new, E_new, F_new = None, None, None, None, None

for i_f, file_i in enumerate(dataset):
    print("Reading file: {}".format(file_i.name))

    lattice, R, z, E, F = None, None, None, None, None

    mols = read(file_i.name, index=':')


    calc = mols[0].get_calculator()


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

    E = np.array([mol.get_potential_energy() for mol in mols])
    F = np.array([mol.get_forces() for mol in mols])


    num_min = np.amin(np.array([R.shape[0], E.shape[0], F.shape[0]]))
    print("\rNumber geometries found: {:,}\n".format(num_min))

    if i_f == 0:
        R_new = R[:num_min].copy()
        E_new = E[:num_min].copy()
        F_new = F[:num_min].copy()
        z_new = z.copy()
    else:
        R_new = np.concatenate((R_new, R[:num_min]), axis=0)
        E_new = np.concatenate((E_new, E[:num_min]), axis=0)
        F_new = np.concatenate((F_new, F[:num_min]), axis=0)

R = R_new.copy()
E = E_new.copy()
F = F_new.copy()
z = z_new.copy()

if len(dataset) > 1:
    print("Final number of concatenated datapoints:", R.shape[0])

print('Please provide a name for this dataset. Otherwise the original filename will be reused.')
custom_name = raw_input('> ').strip()
if custom_name != '':
    name = custom_name

print('Please provide a descriptor for the level of theory used to create this dataset.')
theory = raw_input('> ').strip()
if theory == '':
    theory = 'unknown'

# Base variables contained in every model file.
base_vars = {
    'type': 'd',
    'code_version': __version__,
    'name': name,
    'theory': theory,
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

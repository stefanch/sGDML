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

import numpy as np

from sgdml.utils import io, ui


def raw_input_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print(ui.color_str('[FAIL]', fore_color=ui.RED, bold=True) + ' That is not a valid float.')


# Assumes that the atoms in each molecule are in the same order.
def read_concat_xyz(f):
    n_atoms = None

    R, z = [], []
    for i, line in enumerate(f):
        line = line.strip()
        if not n_atoms:
            n_atoms = int(line)
            print('Number atoms per geometry:      {:>7d}'.format(n_atoms))

        file_i, line_i = divmod(i, n_atoms + 2)

        cols = line.split()
        if line_i >= 2:
            if file_i == 0:  # first molecule
                z.append(io._z_str_to_z_dict[cols[0]])
            R.append(list(map(float, cols[1:4])))

        if file_i % 1000 == 0:
            sys.stdout.write("\rNumber geometries found so far: {:>7d}".format(file_i))
            sys.stdout.flush()
    sys.stdout.write("\rNumber geometries found so far: {:>7d}\n".format(file_i))
    sys.stdout.flush()

    # Only keep complete entries.
    R = R[: int(n_atoms * np.floor(len(R) / float(n_atoms)))]

    R = np.array(R).reshape(-1, n_atoms, 3)
    z = np.array(z)

    f.close()
    return (R, z)


def read_out_file(f, col):

    E = []
    for i, line in enumerate(f):
        line = line.strip()
        if line[0] != '#':  # Ignore comments.
            E.append(float(line.split()[col]))
        if i % 1000 == 0:
            sys.stdout.write("\rNumber lines processed so far:  {:>7d}".format(len(E)))
            sys.stdout.flush()
    sys.stdout.write("\rNumber lines processed so far:  {:>7d}\n".format(len(E)))
    sys.stdout.flush()

    return np.array(E)


parser = argparse.ArgumentParser(
    description='Creates a dataset from extended [TODO] format.'
)
parser.add_argument(
    'geometries',
    metavar='<geometries>',
    type=argparse.FileType('r'),
    help='path to XYZ geometry file',
)
parser.add_argument(
    'forces',
    metavar='<forces>',
    type=argparse.FileType('r'),
    help='path to XYZ force file',
)
parser.add_argument(
    'energies',
    metavar='<energies>',
    type=argparse.FileType('r'),
    help='path to CSV force file',
)
parser.add_argument(
    'energy_col',
    metavar='<energy_col>',
    type=lambda x: io.is_strict_pos_int(x),
    help='which column to parse from energy file (zero based)',
    nargs='?',
    default=0,
)
parser.add_argument(
    '-o',
    '--overwrite',
    dest='overwrite',
    action='store_true',
    help='overwrite existing dataset file',
)
args = parser.parse_args()
geometries = args.geometries
forces = args.forces
energies = args.energies
energy_col = args.energy_col

name = os.path.splitext(os.path.basename(geometries.name))[0]
dataset_file_name = name + '.npz'

dataset_exists = os.path.isfile(dataset_file_name)
if dataset_exists and args.overwrite:
    print(ui.color_str('[INFO]', bold=True) + ' Overwriting existing dataset file.')
if not dataset_exists or args.overwrite:
    print('Writing dataset to \'%s\'...' % dataset_file_name)
else:
    sys.exit(
        ui.color_str('[FAIL]', fore_color=ui.RED, bold=True) + ' Dataset \'%s\' already exists.' % dataset_file_name
    )


print('Reading geometries...')
R, z = read_concat_xyz(geometries)

print('Reading forces...')
F, _ = read_concat_xyz(forces)

print('Reading energies from column %d...' % energy_col)
E = read_out_file(energies, energy_col)

# Prune all arrays to same length.
n_mols = min(min(R.shape[0], F.shape[0]), E.shape[0])
if n_mols != R.shape[0] or n_mols != F.shape[0] or n_mols != E.shape[0]:
    print(
        ui.color_str('[WARN]', fore_color=ui.YELLOW, bold=True)
        + ' Incomplete output detected: Final dataset was pruned to %d points.' % n_mols
    )
R = R[:n_mols, :, :]
F = F[:n_mols, :, :]
E = E[:n_mols]

print(
    ui.color_str('[INFO]', bold=True)
    + ' Geometries, forces and energies must have consistent units.'
)
R_conv_fact = raw_input_float('Unit conversion factor for geometries: ')
R = R * R_conv_fact
F_conv_fact = raw_input_float('Unit conversion factor for forces: ')
F = F * F_conv_fact
E_conv_fact = raw_input_float('Unit conversion factor for energies: ')
E = E * E_conv_fact

# Base variables contained in every model file.
base_vars = {
    'type': 'd',
    'R': R,
    'z': z,
    'E': E[:, None],
    'F': F,
    'name': name,
    'theory': 'unknown',
}
base_vars['md5'] = io.dataset_md5(base_vars)

np.savez_compressed(dataset_file_name, **base_vars)
ui.color_str('[DONE]', fore_color=ui.GREEN, bold=True)

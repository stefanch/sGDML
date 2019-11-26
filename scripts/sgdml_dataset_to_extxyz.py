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

import numpy as np

from sgdml.utils import io, ui


parser = argparse.ArgumentParser(
    description='Converts a native dataset file to extended XYZ format.'
)
parser.add_argument(
    'dataset',
    metavar='<dataset>',
    type=lambda x: io.is_file_type(x, 'dataset'),
    help='path to dataset file',
)
parser.add_argument(
    '-o',
    '--overwrite',
    dest='overwrite',
    action='store_true',
    help='overwrite existing xyz dataset file',
)

args = parser.parse_args()
dataset_path, dataset = args.dataset

name = os.path.splitext(os.path.basename(dataset_path))[0]
dataset_file_name = name + '.xyz'

xyz_exists = os.path.isfile(dataset_file_name)
if xyz_exists and args.overwrite:
    print(ui.color_str('[INFO]', bold=True) + ' Overwriting existing xyz dataset file.')
if not xyz_exists or args.overwrite:
    print(ui.color_str('[INFO]', bold=True) + ' Writing dataset to \'{}\'...'.format(dataset_file_name))
else:
    sys.exit(
        ui.color_str('[FAIL]', fore_color=ui.RED, bold=True) + ' Dataset \'{}\' already exists.'.format(dataset_file_name)
    )

R = dataset['R']
z = dataset['z']
F = dataset['F']

lattice = dataset['lattice'] if 'lattice' in dataset else None

try:
    with open(dataset_file_name, 'w') as file:

        n = R.shape[0]
        for i, r in enumerate(R):

            e = np.squeeze(dataset['E'][i]) if 'E' in dataset else None
            f = dataset['F'][i,:,:]
            ext_xyz_str = io.generate_xyz_str(r, z, e=e, f=f, lattice=lattice) + '\n'

            file.write(ext_xyz_str)

            progr = float(i) / (n - 1)
            ui.progr_bar(i, n - 1, disp_str='Exporting %d data points...' % n)
            
except IOError:
    sys.exit("ERROR: Writing xyz file failed.")

print()

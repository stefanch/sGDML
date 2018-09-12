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

import os, sys
import argparse
import numpy as np

from sgdml.utils import io,ui


parser = argparse.ArgumentParser(description='Converts a native dataset file to extended XYZ format.')
parser.add_argument('dataset', metavar = '<dataset>',\
							 type    = lambda x: ui.is_file_type(x, 'dataset'),\
							 help	 = 'path to dataset file')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing xyz dataset file')

args = parser.parse_args()
dataset_path, dataset = args.dataset

name = os.path.splitext(os.path.basename(dataset_path))[0]
dataset_file_name = name + '.xyz'

xyz_exists = os.path.isfile(dataset_file_name)
if xyz_exists and args.overwrite:	
	print ui.info_str('[INFO]') + ' Overwriting existing xyz dataset file.'
if not xyz_exists or args.overwrite:
	print 'Writing dataset to \'%s\'...' % dataset_file_name
else:
	sys.exit(ui.fail_str('[FAIL]') + ' Dataset \'%s\' already exists.' % dataset_file_name)

z = dataset['z']
R = dataset['R']
F = dataset['F']
E = dataset['E']
n = R.shape[0]

try:
	with open(dataset_file_name, 'w') as f:
		for i,r in enumerate(R):

			f.write(str(len(r)) + '\n' + str(np.squeeze(E[i])))
			for j,atom in enumerate(r):
				f.write('\n' + io._z_to_z_str_dict[z[j]] + '\t')
				f.write('\t'.join(str(x) for x in atom) + '\t')
				f.write('\t'.join(str(x) for x in F[i][j]))
			f.write('\n')

			progr = float(i) / (n-1)
			ui.progr_bar(i, n-1, disp_str='Exporting %d data points...' % n)
except IOError:
	sys.exit("ERROR: Writing xyz file failed.")

print '\n' + ui.pass_str('DONE')
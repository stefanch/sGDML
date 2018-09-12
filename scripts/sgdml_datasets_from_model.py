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


parser = argparse.ArgumentParser(description='Extracts the training and test data subsets from a dataset that were used to construct a model.')
parser.add_argument('model', metavar = '<model_file>',\
							type = lambda x: ui.is_file_type(x, 'model'),\
							help = 'path to model file')
parser.add_argument('dataset', metavar = '<dataset_file>',\
							type = lambda x: ui.is_file_type(x, 'dataset'),\
							help = 'path to dataset file referenced in model')
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help = 'overwrite existing files')
args = parser.parse_args()

model_path, model = args.model
dataset_path, dataset = args.dataset


for s in ['train', 'test']:

	if dataset['md5'] != model[s + '_md5']:
		sys.exit(ui.fail_str('[FAIL]') + ' Dataset fingerprint does not match the one referenced in model for \'%s\'.' % s)

	idxs = model[s + '_idxs']
	R = dataset['R'][idxs,:,:]
	E = dataset['E'][idxs]
	F = dataset['F'][idxs,:,:]

	base_vars = {'type':		'd',\
				 'name':		dataset['name'],\
				 'theory':		dataset['theory'],\
				 'z':			dataset['z'],\
				 'R':			R,\
				 'E':			E,\
				 'F':			F}
	base_vars['md5'] = io.dataset_md5(base_vars)

	subset_file_name = '%s_%s.npz' % (os.path.splitext(os.path.basename(dataset_path))[0], s)
	file_exists = os.path.isfile(subset_file_name)
	if file_exists and args.overwrite:
		print ui.info_str('[INFO]') + ' Overwriting existing model file.'
	if not file_exists or args.overwrite:
		np.savez_compressed(subset_file_name, **base_vars)
		print '[DONE] Extracted %s dataset saved to \'%s\'' % (s,subset_file_name)
	else:
		print ui.warn_str('[WARN]') + ' %s dataset \'%s\' already exists.' % (s.capitalize(), subset_file_name) +\
									  '\n       Run \'python %s -o %s %s\' to overwrite.\n' % (os.path.basename(__file__), model_path, dataset_path)
		sys.exit()
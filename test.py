#!/usr/bin/python

# GDML Force Field
# Author: Stefan Chmiela (stefan@chmiela.com)

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import argparse
import random
import time

import numpy as np

from src.gdml_predict import GDMLPredict
from src.gdml_train import GDMLTrain
from src.utils import io,ui


def batch(iterable, n=1):
	l = len(iterable)
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]

def online_err(err, size, n, mae_n_sum, rmse_n_sum):

	err = np.abs(err)

	mae_n_sum += np.sum(err) / size
	mae = mae_n_sum / n

	rmse_n_sum += np.sum(err**2) / size
	rmse = np.sqrt(rmse_n_sum / n)

	return mae, mae_n_sum, rmse, rmse_n_sum


parser = argparse.ArgumentParser(description='Tests an sGDML model on a given dataset.')
parser.add_argument('model', metavar = '<model>',\
							 type    = lambda x: ui.is_file_type(x, 'model'),\
							 help	 = 'path to model file')
parser.add_argument('dataset', metavar = '<dataset>',\
							   type    = lambda x: ui.is_file_type(x, 'dataset'),\
							   help	   = 'path to dataset file')
parser.add_argument('n_valid', metavar = '<n_valid>',\
							  type    = lambda x: ui.is_strict_pos_int(x),\
							  help    = 'number of validation points from dataset',\
							  nargs   = '?', default = None)
parser.add_argument('-s', '--silent', dest='silent', action='store_true', help = 'suppress output')
parser.add_argument('-u', '--update', dest='update', action='store_true', help = 'update expected prediction errors in model file')
args = parser.parse_args()
model_path, model = args.model
_, dataset = args.dataset
n_valid = args.n_valid

# MODES
# errors nan: test run
# errors set, but n_valid == 0: validation run


e_err = model['e_err'].item()
f_err = model['f_err'].item()

# is this a test or validation run?
is_test = np.isnan(e_err['mae']) and np.isnan(e_err['rmse']) and np.isnan(f_err['mae']) and np.isnan(f_err['rmse'])

if not is_test and not n_valid:
	print ui.warn_str('[WARN]') + ' Skipping already tested model \'%s\'.' % model_path
	sys.exit()

valid_idxs = model['test_idxs']
if not is_test:
	gdml = GDMLTrain()
	valid_idxs = gdml.draw_strat_sample(dataset['E'], n_valid, excl_idxs=np.concatenate([model['train_idxs'], model['test_idxs']]))

#valid_idxs = model['test_idxs']
z = dataset['z']
R_test = dataset['R'][valid_idxs,:,:]
T_test = dataset['E'][valid_idxs]
TG_test = dataset['F'][valid_idxs,:,:]


if not args.silent:
	action_str = 'Test' if is_test else 'Validation'
	print '\n------------------- Model %s -------------------' % action_str
	print '{:<10} {}'.format('Dataset:', model['dataset_theory'])
	print '{:<10} {}\n'.format('%s size:' % action_str, len(valid_idxs))

	if not is_test:
		if model['n_valid'] > 0:
			print 'Expected validation errors (MAE, RMSE):'
		else:
			print 'Recorded test errors (MAE, RMSE):'
		print '| {:<10} {:>.2e}/{:>.2e} kcal/mol'.format('Energy', e_err['mae'], e_err['rmse'])
		print '| {:<10} {:>.2e}/{:>.2e} kcal/mol/Ang'.format('Forces', f_err['mae'], f_err['rmse'])
	print '--------------------------------------------------------\n'

gdml = GDMLPredict(model, batch_size=250, num_workers=4)

n_atoms = z.shape[0]

e_mae_sum, e_rmse_sum = 0,0
f_mae_sum, f_rmse_sum = 0,0
cos_mae_sum, cos_rmse_sum = 0,0
mag_mae_sum, mag_rmse_sum = 0,0

b_size = 1
n_done = 0
t = time.time()
for b_range in batch(range(len(valid_idxs)), b_size):
	n_done += len(b_range)

	r = R_test[b_range].reshape(b_size,-1)
	e = T_test[b_range]
	f = TG_test[b_range].reshape(b_size,-1)

	E,F = gdml.predict(r)

	# energy error
	e_mae, e_mae_sum, e_rmse, e_rmse_sum = online_err(np.squeeze(e) - E, 1, n_done, e_mae_sum, e_rmse_sum)

	# force component error
	f_mae, f_mae_sum, f_rmse, f_rmse_sum = online_err(f.ravel() - F, 3*n_atoms, n_done, f_mae_sum, f_rmse_sum)

	# magnetude error
	F_mags = np.linalg.norm(F.reshape(-1,3), axis=1)
	f_mags = np.linalg.norm(f.reshape(-1,3), axis=1)
	mag_mae, mag_mae_sum, mag_rmse, mag_rmse_sum = online_err(F_mags - f_mags, n_atoms, n_done, mag_mae_sum, mag_rmse_sum)

	# normalized cosine error
	F_norm = F.reshape(-1,3) / F_mags[:,None]
	f_norm = f.reshape(-1,3) / f_mags[:,None]
	cos_err = np.arccos(np.einsum('ij,ij->i', F_norm, f_norm)) / np.pi
	cos_mae, cos_mae_sum, cos_rmse, cos_rmse_sum = online_err(cos_err, n_atoms, n_done, cos_mae_sum, cos_rmse_sum)

	progr = float(n_done) / len(valid_idxs)
	sps = n_done / (time.time() - t) # samples per second
	if args.silent:
		sys.stdout.write('\r \x1b[1;37m[%3d%%]\x1b[0m >> Energy: %.3f/%.3f - Forces: %.3f/%.3f (MAE, RMSE) \x1b[90m@ %.1f geo/s\x1b[0m' % (progr * 100,e_mae,e_rmse,f_mae,f_rmse,sps))
	else:
		sys.stdout.write('\r[%-30s] %03d%% >> Energy: %.3f/%.3f - Forces: %.3f/%.3f (MAE, RMSE) @ %.1f sps' % ('=' * int(progr * 30),progr * 100,e_mae,e_rmse,f_mae,f_rmse,sps))
	sys.stdout.flush()
print ''

e_rmse_pct = ((e_rmse/e_err['rmse'] - 1.) * 100)
f_rmse_pct = ((f_rmse/f_err['rmse'] - 1.) * 100)

if not args.silent:
	print '\nMeasured errors (MAE, RMSE):'
	print '| {:<10} {:>.2e}/{:>.2e} kcal/mol      {:<} '.format('Energy', e_mae, e_rmse, "%s (%+.1f %%)" % ('OK' if e_mae <= e_err['mae'] and e_rmse <= e_err['rmse'] else '!!',e_rmse_pct))
	print '| {:<10} {:>.2e}/{:>.2e} kcal/mol/Ang  {:<} '.format('Forces', f_mae, f_rmse, "%s (%+.1f %%)" % ('OK' if f_mae <= f_err['mae'] and f_rmse <= f_err['rmse'] else '!!',f_rmse_pct))
	print '| {:<10} {:>.2e}/{:>.2e} kcal/mol/Ang'.format('Magnitude', mag_mae, mag_rmse)
	print '| {:<10} {:>.2e}/{:>.2e} [0-1], 0:best'.format('Angle', cos_mae, cos_rmse)
	print ''


model_mutable = dict(model)
model_needs_update = args.update or is_test or model['n_valid'] < len(valid_idxs)

if not is_test:
	model_mutable['n_valid'] = len(valid_idxs)
	if args.update:
		print ' ' + ui.info_str('[INFO]') + ' Errors were updated in model file.'
	elif model['n_valid'] != 0 and model['n_valid'] < len(valid_idxs):
		print ' ' + ui.info_str('[INFO]') + ' Errors were updated in model file because this new validation used more data.'

if model_needs_update:
	model_mutable['e_err'] = {'mae':np.asscalar(e_mae),'rmse':np.asscalar(e_rmse)}
	model_mutable['f_err'] = {'mae':np.asscalar(f_mae),'rmse':np.asscalar(f_rmse)}
	np.savez_compressed(model_path, **model_mutable)
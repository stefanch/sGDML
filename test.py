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
							 type    = lambda x: ui.is_valid_np_file(parser, x),\
							 help	 = 'path to model file')
parser.add_argument('dataset', metavar = '<dataset>',\
							   type    = lambda x: ui.is_valid_np_file(parser, x),\
							   help	   = 'path to dataset file')
parser.add_argument('n_test', metavar = '<n_test>',\
							  type    = lambda x: ui.is_strict_pos_int(x),\
							  help    = 'number of test points from dataset',\
							  nargs   = '?', default = None)
parser.add_argument('-s', '--silent', dest='silent', action='store_true', help = 'suppress output')
parser.add_argument('-u', '--update', dest='update', action='store_true', help = 'update expected prediction errors in model file')
args = parser.parse_args()
model_path, model = args.model
_, dataset = args.dataset
n_test = args.n_test


gdml = GDMLPredict(model, batch_size=250, num_workers=4)

#

#n_train = model['R'].shape[0] # temporal
#n_test += n_train

n_total = len(dataset['T'])
n_test = min(n_test, n_total) if n_test else n_total
test_range = np.random.choice(n_total, n_test, replace=False)


#print 'test_range before'
#print test_range.shape


#X = dataset['R'][test_range,:,:].reshape(n_test,-1)
#Y = model['R'].reshape(model['R'].shape[0],-1)

#alll = []
#for y in Y:

#	w = np.squeeze(np.where(np.isclose(X, y).all(axis=1)))
#	if w:
#		alll.append(w)
#		print w

#test_range = np.delete(test_range,alll, axis=0)




#print test_range.shape
#print 'test_range after'

#sys.exit()


R_test = dataset['R'][test_range,:,:]
z = dataset['z']
T_test = dataset['T'][test_range]
TG_test = dataset['TG'][test_range,:,:]

if not args.silent:
	print
	print '------------------- Model Validation -------------------'
	print "Dataset:    '" + str(model['dataset']) + "'" + (' (' + str(model['theory_level']) + ')' if 'theory_level' in model else '')
	print "Test size:   " + str(n_test)
	print
	print 'Expected prediction errors (MAE, RMSE):'
	print '| {:<10} {:>.3f}/{:>.3f} kcal/mol'.format('Energy', float(model['e_mae']), float(model['e_rmse']))
	print '| {:<10} {:>.3f}/{:>.3f} kcal/mol/Ang'.format('Forces', float(model['f_mae']), float(model['f_rmse']))
	print '--------------------------------------------------------\n'

n_atoms = z.shape[0]

e_mae_sum,e_rmse_sum = [0,0]
f_mae_sum,f_rmse_sum = [0,0]
cos_mae_sum,cos_rmse_sum = [0,0]
mag_mae_sum,mag_rmse_sum = [0,0]

b_size = 1
n_done = 0
t = time.time()
for b_range in batch(range(len(test_range)), b_size):
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

	sys.stdout.write('\r')
	progr = float(n_done) / len(test_range)
	sps = n_done / (time.time() - t) # samples per second
	sys.stdout.write("[%-30s] %03d%% >> Energy: %.3f/%.3f - Forces: %.3f/%.3f (MAE, RMSE) @ %.1f sps" % ('=' * int(progr * 30),progr * 100,e_mae,e_rmse,f_mae,f_rmse,sps))
	sys.stdout.flush()
print ''

e_rmse_pct = ((e_rmse/model['e_rmse'] - 1.) * 100)
f_rmse_pct = ((f_rmse/model['f_rmse'] - 1.) * 100)

if not args.silent: 
	print ''
	print 'Measured prediction errors (MAE, RMSE):'
	print '| {:<10} {:>.3f}/{:>.3f} kcal/mol      {:<} '.format('Energy', e_mae, e_rmse, "%s (%+.1f %%)" % ('OK' if e_mae <= model['e_mae'] and e_rmse <= model['e_rmse'] else '!!',e_rmse_pct))
	print '| {:<10} {:>.3f}/{:>.3f} kcal/mol/Ang  {:<} '.format('Forces', f_mae, f_rmse, "%s (%+.1f %%)" % ('OK' if f_mae <= model['f_mae'] and f_rmse <= model['f_rmse'] else '!!',f_rmse_pct))
	print '| {:<10} {:>.3f}/{:>.3f} kcal/mol/Ang'.format('Magnitude', mag_mae, mag_rmse)
	print '| {:<10} {:>.3f}/{:>.3f} [0-1], 0:best'.format('Angle', cos_mae, cos_rmse)
	print ''

# Update errors if they are not set or on user request.
err_unset = np.isnan(model['e_mae']) and np.isnan(model['e_rmse']) and np.isnan(model['f_mae']) and np.isnan(model['e_rmse'])
if err_unset:
	print '[INFO] Expected prediction errors were recorded in model file.'
elif args.update:
	print '[INFO] Expected prediction errors were updated in model file.'

if args.update or err_unset:
	model_mutable = dict(model)
	model_mutable['n_test'] = n_test
	model_mutable['e_mae'] = np.asscalar(e_mae)
	model_mutable['e_rmse'] = np.asscalar(e_rmse)
	model_mutable['f_mae'] = np.asscalar(f_mae)
	model_mutable['f_rmse'] = np.asscalar(f_rmse)
	np.savez_compressed(model_path, **model_mutable)
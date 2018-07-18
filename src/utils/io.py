import sys
import re
import hashlib
import numpy as np

_z_str_to_z_dict = {'Al':13,'O':8,'N':7,'C':6,'B':5,'H':1}
_z_to_z_str_dict = {v: k for k, v in _z_str_to_z_dict.iteritems()}

def z_str_to_z(z_str):
	return np.array([_z_str_to_z_dict[x] for x in z_str])

def z_to_z_str(z):
	return [_z_to_z_str_dict[int(x)] for x in z]

def task_file_name(task):
	
	n_train = task['R_train'].shape[0]
	n_perms = task['perms'].shape[0]

	dataset = np.squeeze(task['dataset_name'])
	theory_level_str = re.sub('[^\w\-_\.]', '_', str(np.squeeze(task['dataset_theory'])))
	theory_level_str = re.sub('__', '_', theory_level_str)
	sig = np.squeeze(task['sig'])

	return str(n_train) + '-sym' + str(n_perms) + '-sig' + str(sig) + '-' + str(dataset) + '-' + theory_level_str + '.npz'

def dataset_md5(dataset):

	md5_hash = hashlib.md5()
	for key in ['z', 'R', 'E', 'F']:
		md5_hash.update(hashlib.md5(dataset[key].ravel()).digest())

	return md5_hash.hexdigest()

## FILES

# Read of geometry file (xyz format).
def read_geometry(filename):

	try:
		f = open(filename,'r')
	except:
		sys.exit("ERROR: Opening xyz file failed.")

	try:
		lines = f.readlines()
		num_lines = len(lines)

		num_atoms = int(lines[0])
		num_mols = num_lines / (num_atoms + 2)

		R = np.empty([num_atoms, 3, num_mols])
		Z = np.empty([num_atoms, num_mols])

		for i in xrange(0, num_mols):
			blk_start_idx = i * (num_atoms + 2)
			blk_stop_idx = (i+1) * (num_atoms + 2)

			xyz_atoms = [line.split() for line in lines[(blk_start_idx+2):blk_stop_idx]]
			
			R[:,:,i] = np.array([map(float,col[1:4]) for col in xyz_atoms])
			Z[:,i] = np.array([_z_str_to_z_dict[x] for x in [col[0] for col in xyz_atoms]])
	except:
		sys.exit("ERROR: Processing xyz file failed.")
	finally:
		f.close()

	return (R,Z)

# Write geometry file (xyz format).
def write_geometry(filename,r,z,comment_str=''):

	r = np.squeeze(r)

	try:
		with open(filename,'w') as f:
			f.write(str(len(r)) + '\n' + comment_str)
			for i,atom in enumerate(r):
				#print z[i]
				#print _z_to_z_str_dict[int(z[i])]
				f.write('\n' + _z_to_z_str_dict[z[i]] + '\t')
				f.write('\t'.join(str(x) for x in atom))
	except IOError:
		sys.exit("ERROR: Writing xyz file failed.")

# def read_mp_config(filename, dataset):

# 	# Load model.
# 	try:
# 		config = np.load(filename)
# 	except:
# 		print 'Reading config file failed.'
# 		return (-1,125)
# 		#sys.exit("ERROR: Reading config file failed.")

# 	if dataset in config:
# 		return config[dataset]
# 	else:
# 		return (-1,125)

# def write_mp_config(filename, dataset, num_workers, batch_size):

# 	base_vars = {dataset:(num_workers, batch_size)}
# 	np.savez_compressed(filename, **base_vars)
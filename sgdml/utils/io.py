import sys
import re
import hashlib
import numpy as np

_z_str_to_z_dict = {'Au':79,'S':16,'Al':13,'O':8,'N':7,'C':6,'B':5,'H':1}
_z_to_z_str_dict = {v: k for k, v in _z_str_to_z_dict.iteritems()}

def z_str_to_z(z_str):
	return np.array([_z_str_to_z_dict[x] for x in z_str])

def z_to_z_str(z):
	return [_z_to_z_str_dict[int(x)] for x in z]

def train_dir_name(dataset, n_train, is_gdml):
	
	theory_level_str = re.sub('[^\w\-_\.]', '.', str(dataset['theory']))
	theory_level_str = re.sub('\.\.', '.', theory_level_str)
	return 'tmp/%s-%s-train%d%s' % (dataset['name'], theory_level_str, n_train, '-gdml' if is_gdml else '')

def task_file_name(task):
	
	n_train = task['train_idxs'].shape[0]
	n_perms = task['perms'].shape[0]
	sig = np.squeeze(task['sig'])

	return 'task-train%d-sym%d-sig%d.npz' % (n_train, n_perms, sig)

def model_file_name(task_or_model, is_extended=False):
	
	n_train = task_or_model['train_idxs'].shape[0]
	n_perms = task_or_model['perms'].shape[0]
	sig = np.squeeze(task_or_model['sig'])

	if is_extended:
		dataset = np.squeeze(task_or_model['dataset_name'])
		theory_level_str = re.sub('[^\w\-_\.]', '.', str(np.squeeze(task_or_model['dataset_theory'])))
		theory_level_str = re.sub('\.\.', '.', theory_level_str)
		return '%s-%s-train%d-sym%d.npz' % (dataset, theory_level_str, n_train, n_perms)
	return 'model-train%d-sym%d-sig%d.npz' % (n_train, n_perms, sig)

def dataset_md5(dataset):

	md5_hash = hashlib.md5()
	for key in ['z', 'R', 'E', 'F']:
		md5_hash.update(hashlib.md5(dataset[key].ravel()).digest())

	return md5_hash.hexdigest()

## FILES

# Read geometry file (xyz format).
# R: (n_geo,3*n_atoms)
# z: (3*n_atoms,)
def read_xyz(file_path):

	with open(file_path,'r') as f:
		n_atoms = None

		R,z = [],[]
		for i,line in enumerate(f):
			line = line.strip()
			if not n_atoms:
				n_atoms = int(line)

			cols = line.split()
			file_i, line_i = divmod(i, n_atoms+2)
			if line_i >= 2:
				R.append(map(float,cols[1:4]))
				if file_i == 0: # first molecule
					z.append(_z_str_to_z_dict[cols[0]])

		R = np.array(R).reshape(-1,3*n_atoms)
		z = np.array(z)

		f.close()
	return R,z

# Write geometry file (xyz format).
def write_geometry(filename,r,z,comment_str=''):

	r = np.squeeze(r)
	try:
		with open(filename,'w') as f:
			f.write(str(len(r)) + '\n' + comment_str)
			for i,atom in enumerate(r):
				f.write('\n' + _z_to_z_str_dict[z[i]] + '\t')
				f.write('\t'.join(str(x) for x in atom))
	except IOError:
		sys.exit("ERROR: Writing xyz file failed.")
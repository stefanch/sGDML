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

# Read geometry file (xyz format).
# def read_geometry(filename):

# 	try:
# 		f = open(filename,'r')
# 	except:
# 		sys.exit("ERROR: Opening xyz file failed.")

# 	try:
# 		lines = f.readlines()
# 		num_lines = len(lines)

# 		num_atoms = int(lines[0])
# 		num_mols = num_lines / (num_atoms + 2)

# 		R = np.empty([num_atoms, 3, num_mols])
# 		Z = np.empty([num_atoms, num_mols])

# 		for i in xrange(0, num_mols):
# 			blk_start_idx = i * (num_atoms + 2)
# 			blk_stop_idx = (i+1) * (num_atoms + 2)

# 			xyz_atoms = [line.split() for line in lines[(blk_start_idx+2):blk_stop_idx]]
			
# 			R[:,:,i] = np.array([map(float,col[1:4]) for col in xyz_atoms])
# 			Z[:,i] = np.array([_z_str_to_z_dict[x] for x in [col[0] for col in xyz_atoms]])
# 	except:
# 		sys.exit("ERROR: Processing xyz file failed.")
# 	finally:
# 		f.close()

# 	return (R,Z)

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
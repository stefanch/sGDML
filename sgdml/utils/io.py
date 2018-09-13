import sys
import re
import hashlib
import numpy as np

_z_str_to_z_dict = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,'Bk':97,'Cf':98,'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103,'Rf':104,'Db':105,'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112,'Uuq':114,'Uuh':116}
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
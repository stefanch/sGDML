import os

#import os, sys
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
#sys.path.append(BASE_DIR)

import argparse

import numpy as np
import scipy.io


def yes_or_no(question):
	reply = str(raw_input(question+' (y/n): ')).lower().strip()
	if not reply or reply[0] != 'y':
		return False
	else:
		return True


# COLORS

def white_back_str(str):
	return '\x1b[1;7m' + str + '\x1b[0m'

def white_bold_str(str):
	return '\x1b[1;37m' + str + '\x1b[0m'

def gray_str(str):
	return '\x1b[90m' + str + '\x1b[0m'

def underline_str(str):
	return '\x1b[4m' + str + '\x1b[0m'

def blink_str(str):
	return '\x1b[5m' + str + '\x1b[0m'

def info_str(str):
	return str

def pass_str(str):
	return '\x1b[1;32m' + str + '\x1b[0m'

def warn_str(str):
	return '\x1b[1;33m' + str + '\x1b[0m'

def fail_str(str):
	return '\x1b[1;31m' + str + '\x1b[0m'


# USER INPUT VALIDATION

#def is_valid_np_file(parser, arg):
#	try:
#		return arg, np.load(arg)
#	except:
#		parser.error("Reading '%s' failed." % arg)

def is_file_type(arg, type):

	if not arg.endswith('.npz'):
		argparse.ArgumentTypeError('{0} is not a .npz file'.format(arg))

	try:
		file = np.load(arg)
	except:
		raise argparse.ArgumentTypeError('{0} is not readable'.format(arg))
			
	if 'type' not in file or file['type'] != type[0]:
		raise argparse.ArgumentTypeError('{0} is not a {1} file'.format(arg,type))

	return arg, file


#def is_valid_mat_file(parser, arg):
#	try:
#		return arg, scipy.io.loadmat(arg)
#	except:
#		parser.error("Reading '%s' failed." % arg)

#def is_dir(arg):
#	if not os.path.isdir(arg):
#		raise argparse.ArgumentTypeError("{0} is not a directory".format(arg))
#	else:
#		return arg

# if file is provided, this function acts like its a directory with just one file
def is_dir_with_file_type(arg, type, or_file=False):

	if or_file and os.path.isfile(arg): # arg: file path
		is_file_type(arg, type) # raises exception if there is a problem with file
		file_name = os.path.basename(arg)
		file_dir = os.path.dirname(arg)
		return file_dir, [file_name]
	else: # arg: dir

		if not os.path.isdir(arg):
			raise argparse.ArgumentTypeError("{0} is not a directory".format(arg))

		file_names = []
		for file_name in sorted(os.listdir(arg)):
			if file_name.endswith('.npz'):
				file_path = os.path.join(arg, file_name)
				try:
					file = np.load(file_path)
				except:
					raise argparse.ArgumentTypeError('{0} contains unreadable .npz files'.format(arg))
					
				if 'type' in file and file['type'] == type[0]:
					file_names.append(file_name)

		if not len(file_names):
			raise argparse.ArgumentTypeError('{0} contains no {1} files'.format(arg, type))

		return arg, file_names

def is_strict_pos_int(arg):
	x = int(arg)
	if x <= 0:
		raise argparse.ArgumentTypeError('Parameter must be >0.')
	return x

#!/usr/bin/python

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import re
import urllib2

from src.utils import io, ui

base_url = 'http://www.quantum-machine.org/gdml/datasets/npz/'
target_dir = BASE_DIR + '/datasets/npz/'

print "Contacting server (%s)..." % (base_url)
response = urllib2.urlopen(base_url + 'list.php')
line = response.readlines()

print ''
print 'Available datasets:'

print '{:<2} {:<25} {:>4}'.format('ID', 'Dataset', 'Size')
print '-'*36

datasets = line[0].split(';')
for i,dataset in enumerate(datasets):
	name, size = dataset.split(',')
	size = int(size) / 1024**2 # Bytes to MBytes

	print '{:>2d} {:<25} {:>4d} MB'.format(i, name, size)
print ''

down_list = raw_input('Please list which datasets to download (e.g. 0 1 2 6) or type \'all\': ')
down_idxs = []
if 'all' in down_list.lower():
	down_idxs = range(len(datasets))
elif re.match("^ *[0-9][0-9 ]*$", down_list): # only digits and spaces, at least one digit
	down_idxs = [int(idx) for idx in re.split(r'\s+', down_list.strip())]
	down_idxs = list(set(down_idxs))
else:
	print ' ABORTED.'

if not os.path.exists(target_dir):
	os.makedirs(target_dir)

for idx in down_idxs:
	if not idx in range(len(datasets)):
		print 'Index ' + idx + ' out of range, skipping.'
	else:
		name = datasets[idx].split(',')[0]
		if os.path.exists(target_dir + name):
			print "'%s' exists, skipping." % (name)
			continue

		request = urllib2.urlopen(base_url + name)
		file = open(target_dir + name, 'wb')
		filesize = int(request.info().getheaders("Content-Length")[0])

		print "Downloading: [%d] %s (%s bytes)" % (idx, name, filesize)
		size = 0
		block_sz = 1024
		while True:
			buffer = request.read(block_sz)
			if not buffer:
				break
			size += len(buffer)
			file.write(buffer)

			sys.stdout.write('\r')
			progr = float(size) / float(filesize)
			sys.stdout.write("[%-30s] %03d%%" % ('=' * int(progr * 30),progr * 100))
			sys.stdout.flush()
		print ''
		file.close()
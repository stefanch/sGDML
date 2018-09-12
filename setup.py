import os
from setuptools import setup

from os import path
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md')) as f:
	long_description = f.read()

# Scripts
scripts = []
for dirname, dirnames, filenames in os.walk('scripts'):
	for filename in filenames:
		if filename.endswith('.py'):
			scripts.append(os.path.join(dirname, filename))

setup(name = 'sgdml',
			version = '0.7.0.dev0',
			description = 'Reference implementation of the GDML and sGDML force field models.',
			long_description = long_description,
			long_description_content_type = 'text/markdown',
			classifiers=[
					'Development Status :: 4 - Beta',
					'Environment :: Console',
					'Intended Audience :: Science/Research'
					'Intended Audience :: Education'
					'Intended Audience :: Developers'  
					'License :: OSI Approved :: MIT License'
					'Operating System :: MacOS :: MacOS X',
					'Operating System :: POSIX',
					'Programming Language :: Python :: 2 :: Only',
					'Topic :: Scientific/Engineering :: Chemistry',
					'Topic :: Scientific/Engineering :: Physics',
					'Topic :: Software Development :: Libraries :: Python Modules'],
			url = 'http://www.gdml.ml',
			author = 'Stefan Chmiela',
			author_email = 'noreply@chmiela.com',
			license = 'MIT',
			packages = ['sgdml'],
			install_requires = [
					'numpy >= 1.13.0',
					'scipy >= 1.1.0'],
			entry_points={
					'console_scripts': ['sgdml=sgdml.cli:main']},
			scripts=scripts,
			include_package_data=True,
			zip_safe=False)
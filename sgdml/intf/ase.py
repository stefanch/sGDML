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

import numpy as np

from sgdml.predict import GDMLPredict

from ase.calculators.calculator import Calculator
from ase.units import mol, kcal


class SGDMLCalculator(Calculator):

	implemented_properties = ['energy', 'forces']

	def __init__(self, model_path, *args, **kwargs):

		super(SGDMLCalculator, self).__init__(*args, **kwargs)

		model = np.load(model_path)
		self.gdml_predict = GDMLPredict(model)

		self.from_kcal_mol = kcal/mol

	def calculate(self, atoms=None, *args, **kwargs):

		super(SGDMLCalculator, self).calculate(atoms, *args, **kwargs)

		r = np.array(atoms.get_positions())
		e,f = self.gdml_predict.predict(r.ravel())
		
		# kcal/mol to eV
		e *= self.from_kcal_mol
		f *= self.from_kcal_mol

		self.results = {'energy': e, 'forces': f.reshape(-1, 3)}
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

import logging
import numpy as np

from ase.calculators.calculator import Calculator
from ase.units import kcal, mol

from ..predict import GDMLPredict


class SGDMLCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(
        self, model_path, E_to_eV=kcal / mol, F_to_eV_Ang=kcal / mol, *args, **kwargs
    ):
        """
        ASE calculator for the sGDML force field.

        A calculator takes atomic numbers and atomic positions from an Atoms object and calculates the energy and forces.

        Note
        ----
        ASE uses eV and Angstrom as energy and length unit, respectively. Unless the paramerters `E_to_eV` and `F_to_eV_Ang` are specified, the sGDML model is assumed to use kcal/mol and Angstorm and the appropriate conversion factors are set accordingly.
        Here is how to find them: `ASE units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

        Parameters
        ----------
                model_path : :obj:`str`
                        Path to a sGDML model file
                E_to_eV : float, optional
                        Conversion factor from whatever energy unit is used by the model to eV. By default this parameter is set to convert from kcal/mol.
                F_to_eV_Ang : boolean, optional
                        Conversion factor from whatever length unit is used by the model to Angstrom. By default, the length unit is not converted.
        """

        super(SGDMLCalculator, self).__init__(*args, **kwargs)

        self.log = logging.getLogger(__name__)

        model = np.load(model_path)
        self.gdml_predict = GDMLPredict(model)
        # self.gdml_predict.prepare_parallel()

        self.log.warning(
            'Please remember to specify the proper conversion factors, if your model does not use \'kcal/mol\' and \'Ang\' as units.'
        )

        # Converts energy from the unit used by the sGDML model to eV.
        self.E_to_eV = E_to_eV

        # Converts force from the unit used by the sGDML model to eV/Ang.
        self.F_to_eV_Ang = F_to_eV_Ang

    def calculate(self, atoms=None, *args, **kwargs):

        super(SGDMLCalculator, self).calculate(atoms, *args, **kwargs)

        r = np.array(atoms.get_positions())
        e, f = self.gdml_predict.predict(r.ravel())

        # convert model units to ASE default units (eV and Ang)
        e *= self.E_to_eV
        f *= self.F_to_eV_Ang

        self.results = {'energy': e, 'forces': f.reshape(-1, 3)}

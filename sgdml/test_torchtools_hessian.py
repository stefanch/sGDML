#!/usr/bin/env python
# coding: utf-8
"""
test torch implementation of machine-learned sGDML model (energies, gradients and Hessians)
"""
model_file = 'models/ethanol.npz'
geometry_file = 'geometries/ethanol.xyz'

import numpy as np
import numpy.linalg as la
import torch
import logging
import time

from sgdml.utils import io

from sgdml.torchtools_hessian import  GDMLPredict
from sgdml.torchtools import GDMLTorchPredict

# # Logging
logger = logging.getLogger(__name__)

# GPU or CUDA?
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    logger.info("CUDA available")
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
    
# load model fitted to ground state forces
model = np.load(model_file, allow_pickle=True)
# reference implementation
gdml_ref = GDMLTorchPredict(model)
# new implementation with analytical Hessians
gdml = GDMLPredict(model).to(device)
# load geometry
r,_ = io.read_xyz(geometry_file)
coords = torch.from_numpy(r).to(device)

##########################################################################
#                                                                        #
# check that energies and gradients agree with reference implementation  #
#                                                                        #
##########################################################################

# make random numbers reproducible
torch.manual_seed(0)

natom = coords.size()[1]//3
# timings for different batch sizes
for batch_size in [1,10,100,1000]:
    logger.info(f"batch size {batch_size}")
    # batch (B,3*N)
    rs = coords.repeat(batch_size, 1) + 0.1 * torch.rand(batch_size,3*natom).to(device)
    # (B, N, 3)
    rs_3N = rs.reshape(batch_size, -1, 3)

    t_start = time.time()
    # compute energy and Hessian with reference implementation
    en_ref, force_ref = gdml_ref.forward(rs_3N)
    grad_ref = -force_ref.reshape(rs.size())
            
    t_end = time.time()
    logger.info(f"timing reference implementation, energy+gradient    : {t_end-t_start} seconds")

    t_start = time.time()
    # and compare with new implementation
    en, grad, hessian = gdml.forward(rs)

    t_end = time.time()
    logger.info(f"timing new implementation, energy+gradient+hessian  : {t_end-t_start} seconds")
            
    # error per sample
    err_en = torch.norm(en_ref - en)/batch_size
    err_grad = torch.norm(grad_ref - grad)/batch_size
    
    logger.info(f"   error of energy   : {err_en}")
    logger.info(f"   error of gradient : {err_grad}")
            
    assert err_en < 1.0e-4
    assert err_grad < 1.0e-4

###############################################################
#                                                             #
# compare numerical and analytic Hessians of sGDML potential  #
#                                                             #
###############################################################
from sgdml.intf.ase_calc import SGDMLCalculator
from ase.io.xyz import read_xyz
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.units import kcal, mol

# compute Hessian numerically using ASE
with open(geometry_file) as f:
    molecule = next(read_xyz(f))
sgdml_calc = SGDMLCalculator(model_file)
molecule.calc = sgdml_calc

# optimization
opt = BFGS(molecule)
opt.run(fmax=0.001)
# optimized geometry
coords_opt = torch.from_numpy(molecule.get_positions()).reshape(1,-1).to(device)

# frequencies
vib = Vibrations(molecule, name="/tmp/vib_sgdml")
vib.run()
vib.get_energies()
vib.clean()

# convert numerical Hessian from eV Ang^{-2} to kcal/mol Ang^{-2}
hessian_numerical = vib.H / (kcal / mol)


# compute analytic Hessian directly from sGDML model
hessian_analytical = gdml.forward(coords_opt)[2][0,:,:].cpu().numpy()

# check that Hessian is symmetric
err_sym = la.norm(hessian_analytical - hessian_analytical.T)
logger.info(f"|Hessian-Hessian^T|= {err_sym}")
assert err_sym < 1.0e-8

"""
# compare Hessians visually
import matplotlib.pyplot as plt
ax1 = plt.subplot(1,3,1)
ax1.set_title("numerical Hessian")
ax1.imshow(hessian_numerical)

ax2 = plt.subplot(1,3,2)
ax2.set_title("analytical Hessian")
ax2.imshow(hessian_analytical)

ax3 = plt.subplot(1,3,3)
ax3.set_title("difference")
ax3.imshow(hessian_numerical - hessian_analytical)

plt.show()
"""

# check that numerical and analytical Hessians agree within numerical errors
err = la.norm(hessian_numerical - hessian_analytical)/la.norm(hessian_numerical)
logger.info(f"|Hessian(num)-Hessian(ana)|/|Hessian(num)|= {err}")
assert err < 1.0e-3

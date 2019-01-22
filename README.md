# Symmetric Gradient Domain Machine Learning (sGDML)

For more details visit: [http://quantum-machine.org/gdml/doc/](http://quantum-machine.org/gdml/doc/)

#### Requirements:
- Python 2.7
- NumPy (>=1.13.0)
- SciPy

## Getting started

### Stable release

Most systems come with the default package manager for Python ``pip`` already preinstalled. We install ``sgdml`` by simply calling:

`pip install sgdml`

The ``sgdml`` command-line interface and the corresponding Python API can now be used from anywhere on the system.

### Development version

#### Clone the repository

`git clone https://github.com/stefanch/sGDML.git`

`cd sGDML`

...or update your existing local copy with

`git pull origin master`

##### Install

`pip install -e .`

Using the flag ``--user``, we can tell ``pip`` to install the package to the current users's home directory, instead of system-wide. This option might require you to update your system's ``PATH`` variable accordingly.

## Reconstruct your first force field

Download one of the example datasets:

`sgdml-get dataset ethanol`

Train a force field model:

`sgdml all ethanol.npz 200 1000 5000`

## Query a force field

```python
import numpy as np
from sgdml.predict import GDMLPredict
from sgdml.utils import io

r,_ = io.read_xyz('examples/geometries/ethanol.xyz') # 9 atoms
print r.shape # (1,27)

model = np.load('models/ethanol.npz')
gdml = GDMLPredict(model)
e,f = gdml.predict(r)
print e.shape # (1,)
print f.shape # (1,27)
```

## References

* [1] Chmiela, S., Tkatchenko, A., Sauceda, H. E., Poltavsky, I., Schütt, K. T., Müller, K.-R.,
*Machine Learning of Accurate Energy-conserving Molecular Force Fields.*
Science Advances, 3(5), e1603015 (2017)   
[10.1126/sciadv.1603015](http://dx.doi.org/10.1126/sciadv.1603015)

* [2] Chmiela, S., Sauceda, H. E., Müller, K.-R., & Tkatchenko, A.,
*Towards Exact Molecular Dynamics Simulations with Machine-Learned Force Fields.*
Nature Communications, 9(1), 3887 (2018)   
[10.1038/s41467-018-06169-2](https://doi.org/10.1038/s41467-018-06169-2)

* [3] Chmiela, S., Sauceda, H. E., Poltavsky, I., Müller, K.-R., & Tkatchenko, A.,
*sGDML: Constructing Accurate and Data Efficient Molecular Force Fields Using Machine Learning.*
[arXiv:1812.04986](https://arxiv.org/abs/1812.04986)
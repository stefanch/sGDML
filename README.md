# Symmetric Gradient Domain Machine Learning (sGDML)

For more details visit: [sgdml.org](http://sgdml.org/)  
Documentation can be found here: [docs.sgdml.org](http://docs.sgdml.org/)

#### Requirements:
- Python 3.7+
- PyTorch (>=1.8)
- NumPy (>=1.19)
- SciPy (>=1.1)

#### Optional:
- ASE (>=3.16.2) (to run atomistic simulations)

## Getting started

### Stable release

Most systems come with the default package manager for Python ``pip`` already preinstalled. Install ``sgdml`` by simply calling:

```
$ pip install sgdml
```

The ``sgdml`` command-line interface and the corresponding Python API can now be used from anywhere on the system.

### Development version

#### (1) Clone the repository

```
$ git clone https://github.com/stefanch/sGDML.git
$ cd sGDML
```

...or update your existing local copy with

```
$ git pull origin master
```

#### (2) Install

```
$ pip install -e .
```

Using the flag ``--user``, you can tell ``pip`` to install the package to the current users's home directory, instead of system-wide. This option might require you to update your system's ``PATH`` variable accordingly.


### Optional dependencies

Some functionality of this package relies on third-party libraries that are not installed by default. These optional dependencies (or "package extras") are specified during installation using the "square bracket syntax":

```
$ pip install sgdml[<optional1>]
```

#### Atomic Simulation Environment (ASE)

If you are interested in interfacing with [ASE](https://wiki.fysik.dtu.dk/ase/) to perform atomistic simulations (see [here](http://docs.sgdml.org/applications.html) for examples), use the ``ase`` keyword:

```
$ pip install sgdml[ase]
```

## Reconstruct your first force field

Download one of the example datasets:

```
$ sgdml-get dataset ethanol_dft
```

Train a force field model:

```
$ sgdml all ethanol_dft.npz 200 1000 5000
```

## Query a force field

```python
import numpy as np
from sgdml.predict import GDMLPredict
from sgdml.utils import io

r,_ = io.read_xyz('geometries/ethanol.xyz') # 9 atoms
print(r.shape) # (1,27)

model = np.load('models/ethanol.npz')
gdml = GDMLPredict(model)
e,f = gdml.predict(r)
print(e.shape) # (1,)
print(f.shape) # (1,27)
```

## Authors

* Stefan Chmiela
* Jan Hermann

We appreciate and welcome contributions and would like to thank the following people for participating in this project:

* Huziel Sauceda
* Igor Poltavsky
* Luis Gálvez
* Danny Panknin
* Grégory Fonseca

## References

* [1] Chmiela, S., Tkatchenko, A., Sauceda, H. E., Poltavsky, I., Schütt, K. T., Müller, K.-R.,
*Machine Learning of Accurate Energy-conserving Molecular Force Fields.*
Science Advances, 3(5), e1603015 (2017)   
[10.1126/sciadv.1603015](http://dx.doi.org/10.1126/sciadv.1603015)

* [2] Chmiela, S., Sauceda, H. E., Müller, K.-R., Tkatchenko, A.,
*Towards Exact Molecular Dynamics Simulations with Machine-Learned Force Fields.*
Nature Communications, 9(1), 3887 (2018)   
[10.1038/s41467-018-06169-2](https://doi.org/10.1038/s41467-018-06169-2)

* [3] Chmiela, S., Sauceda, H. E., Poltavsky, I., Müller, K.-R., Tkatchenko, A.,
*sGDML: Constructing Accurate and Data Efficient Molecular Force Fields Using Machine Learning.*
Computer Physics Communications, 240, 38-45 (2019)
[10.1016/j.cpc.2019.02.007](https://doi.org/10.1016/j.cpc.2019.02.007)

* [4] Chmiela, S., Vassilev-Galindo, V., Unke, O. T., Kabylda, A., Sauceda, H. E., Tkatchenko, A., Müller, K.-R.,
*Accurate global machine learning force fields for molecules with hundreds of atoms*
Preprint (2022)
[arXiv:2209.14865](https://arxiv.org/abs/2209.14865)
# (Symmetric) Gradient Domain Machine Learning (sGDML)

#### Requirements:
- Python 2.7
- NumPy (>=1.13.0)
- SciPy

## Getting started

#### Clone the repository

`git clone https://github.com/stefanch/sGDML.git`

`cd sGDML`

##### ...or update your local copy

`git pull origin master`

#### Reconstruct your first force field

`python tools/download_datasets.py`

`python assist.py train datasets/npz/ethanol.npz 200 1000 5000`

#### Query a force field

```python
import sys
import numpy as np
from src.gdml_predict import GDMLPredict
from src.utils import io

r,_ = io.read_xyz('examples/geometries/ethanol.xyz') # 9 atoms
print r.shape # (1,27)

model = np.load('models/ethanol.npz')
gdml = GDMLPredict(model)
e,f = gdml.predict(r)
print e.shape # ()
print f.shape # (1,27)
```

## References

* [1] Chmiela, S., Tkatchenko, A., Sauceda, H. E., Poltavsky, Igor, Schütt, K. T., Müller, K.-R.,
*Machine Learning of Accurate Energy-conserving Molecular Force Fields.*
Science Advances, 3(5), e1603015 (2017)   
[10.1126/sciadv.1603015](http://dx.doi.org/10.1126/sciadv.1603015)

* [2] Chmiela, S., Sauceda, H., Müller, K.-R., & Tkatchenko, A.,
*Towards Exact Molecular Dynamics Simulations with Machine-Learned Force Fields.*
arXiv preprint, 1802.09238 (2018)   
[arXiv:1802.09238](https://arxiv.org/abs/1802.09238)
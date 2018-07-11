# (Symmetric) Gradient Domain Machine Learning (sGDML)

#### Requirements:
- Python 2.7
- NumPy (>=1.13.0)
- SciPy

## Getting started

#### Clone the repository

`git clone https://gitlab.tubit.tu-berlin.de/chmidhkg/sGDML.git`

`cd sGDML`

##### ...or update your local copy

`git pull origin master`

#### Reconstruct your first force field

`python tools/download_datasets.py`

`python train_assist.py datasets/npz/ethanol.npz 200 1000`

## References

* [1] Chmiela, S., Tkatchenko, A., Sauceda, H. E., Poltavsky, Igor, Schütt, K. T., Müller, K.-R.,
*Machine Learning of Accurate Energy-conserving Molecular Force Fields.*
Science Advances, 3(5), e1603015 (2017)   
[10.1126/sciadv.1603015](http://dx.doi.org/10.1126/sciadv.1603015)

* [2] Chmiela, S., Sauceda, H., Müller, K.-R., & Tkatchenko, A.,
*Towards Exact Molecular Dynamics Simulations with Machine-Learned Force Fields.*
arXiv preprint, 1802.09238 (2018)   
[arXiv:1802.09238](https://arxiv.org/abs/1802.09238)
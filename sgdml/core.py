try:
    from ase.io import read
except ImportError:
    raise ImportError('Optional ASE dependency not found! Please run \'pip install sgdml[ase]\' to install it.')
import numpy as np
import os
import sgdml
from functools import partial
from . import __version__, MAX_PRINT_WIDTH, DONE, NOT_DONE
from .utils import ui, io, perm, desc
from .cli import _batch, _online_err
from .train import GDMLTrain
from .predict import GDMLPredict
import timeit
import time


def dummy_callback(done=None, dummy=None, **kwargs):
    pass


class AssistantError(Exception):
    pass


class Dataset:
    """
        Dataset object

        This object is a custom dataset holder for the sGDML. both object API and
        directory API for compatibility with the CLI
        """
    def __init__(self, dataset_path: str, to_file=False,
                 name=None,
                 theory='unknown',
                 overwrite=False,
                 r_unit='', e_unit='',
                 verbose=True):

        """
        Constructor for the Data object.
        using ase.io.read as a general file format loader.

        Parameters
        ----------
            dataset_path : str
                Path to dataset file.
            to_file : bool, optional
                True: save Dataset object to file
                False: only return the object
            name : str, optional
                A custom name to the dataset
            theory : str, optional
                The source method for the dataset
            overwrite : bool, optional
                True: rewrite existing dataset file if present
                False: will abort if existing dataset file is present
            r_unit : str, optional
                Distance units of the Dataset
            e_unit : str, optional
                Energy units of the Dataset
            verbose : bool, optional
                True: will display the process information
                False: will not display any informtion


        Returns
        -------
            sgdml.core.Dataset
                 type :obj:`sgdml.core.Dataset`.
                        """

        self.type = 'd'
        self.code_version = sgdml.__version__
        self.name = alias_str(name if not name is None else os.path.splitext(os.path.basename(dataset_path))[0])
        self.theory = alias_str(theory)
        mols = read(dataset_path, index=':')

        dataset_file_name = self.name + '.npz'
        dataset_exists = os.path.isfile(dataset_file_name)
        if (dataset_exists and overwrite) and verbose:
            print(ui.color_str('[INFO]', bold=True) + ' Overwriting existing dataset file.')
        if (not dataset_exists or overwrite):
            if verbose:
                # print('Writing dataset to \'{}\'...'.format(dataset_file_name))
                pass
        else:
            if verbose:
                print(ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
                      + ' Dataset \'{}\' already exists.'.format(dataset_file_name))
            return

        # filter incomplete outputs from trajectory
        mols = [mol for mol in mols if mol.get_calculator() is not None]

        lattice, R, z, E, F = None, None, None, None, None

        calc = mols[0].get_calculator()
        if verbose:
            print("\rNumber geometries: {:,}".format(len(mols)))
            print("\rAvailable properties: " + ', '.join(calc.results))
            print()

        if 'forces' not in calc.results:
            if verbose:
                print(ui.color_str('[FAIL]', fore_color=ui.RED, bold=True) + ' Forces are missing in the input file!')
            return

        lattice = np.array(mols[0].get_cell())
        if not np.any(lattice):
            if verbose:
                print(ui.color_str('[INFO]', bold=True) + ' No lattice vectors specified.')

        Z = np.array([mol.get_atomic_numbers() for mol in mols])
        all_z_the_same = (Z == Z[0]).all()
        if not all_z_the_same:
            if verbose:
                print(ui.color_str('[FAIL]', fore_color=ui.RED, bold=True) + ' Order of atoms changes accross dataset.')
            return

        lattice = np.array(mols[0].get_cell())
        if not np.any(lattice):  # all zeros
            lattice = None

        R = np.array([mol.get_positions() for mol in mols])
        z = Z[0]

        E = np.array([mol.get_potential_energy() for mol in mols])
        F = np.array([mol.get_forces() for mol in mols])

        self.F_min, self.F_max = np.min(F.ravel()), np.max(F.ravel())
        self.F_mean, self.F_var = np.mean(F.ravel()), np.var(F.ravel())
        if r_unit != '':
            self.r_unit = r_unit
        if e_unit != '':
            self.e_unit = e_unit

        if E is not None:
            self.E = E
            self.E_min, self.E_max = np.min(E), np.max(E)
            self.E_mean, self.E_var = np.mean(E), np.var(E)
        else:
            if verbose:
                print(ui.color_str('[INFO]', bold=True) + ' No energy labels found in dataset.')
        self.z = z
        self.R = R
        self.F = F
        if lattice is not None:
            self.lattice = lattice

        self.md5 = io.dataset_md5(self)
        if to_file:
            self.dataset_file_name = dataset_file_name
            np.save(dataset_file_name, mols)
            if verbose:
                print(ui.color_str('[INFO]', bold=True) + f' save dataset to {dataset_file_name}')

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __str__(self, title_str='Dataset properties'):
        text = ''
        text += ui.white_bold_str(title_str) + '\n'

        n_mols, n_atoms, _ = self['R'].shape
        text += '  {:<18} {} ({:<d} atoms)'.format('Name:', self['name'], n_atoms) + '\n'
        text += '  {:<18} {}'.format('Theory:', self['theory']) + '\n'
        text += '  {:<18} {:,} data points'.format('Size:', n_mols) + '\n'

        lat_str = 'n/a'
        if self['lattice'] is not None:
            lat_str = ui.gen_lattice_str(self['lattice'])
            lengths, angles = io.lattice_vec_to_par(self['lattice'])

        text += '  {:<18} {}'.format('Lattice:', lat_str) + '\n'
        if self['lattice'] is not None:
            text += '    {:<16} a = {:g}, b = {:g}, c = {:g}'.format('Lengths:', *lengths) + '\n'
            text += '    {:<16} alpha = {:g}, beta = {:g}, gamma = {:g}'.format('Angles [deg]:', *angles) + '\n'

        if 'E' in self:

            e_unit = 'unknown unit'
            if 'e_unit' in self:
                e_unit = self['e_unit']

            text += '  Energies [{}]:'.format(e_unit)
            if 'E_min' in self and 'E_max' in self:
                E_min, E_max = self['E_min'], self['E_max']
            else:
                E_min, E_max = np.min(self['E']), np.max(self['E'])
            E_range_str = ui.gen_range_str(E_min, E_max)
            text += '    {:<16} {}'.format('Range:', E_range_str) + '\n'

            E_mean = self['E_mean'] if 'E_mean' in self else np.mean(self['E'])
            text += '    {:<16} {:<.3f}'.format('Mean:', E_mean) + '\n'

            E_var = self['E_var'] if 'E_var' in self else np.var(self['E'])
            text += '    {:<16} {:<.3f}'.format('Variance:', E_var) + '\n'
        else:
            text += '  {:<18} {}'.format('Energies:', 'n/a') + '\n'

        f_unit = 'unknown unit'
        if 'r_unit' in self and 'e_unit' in self:
            f_unit = (ui.unicode_str(self['e_unit']) + '/' + ui.unicode_str(self['r_unit']))

        text += '  Forces [{}]:'.format(f_unit) + '\n'

        if 'F_min' in self and 'F_max' in self:
            F_min, F_max = self['F_min'], self['F_max']
        else:
            F_min, F_max = np.min(self['F'].ravel()), np.max(self['F'].ravel())
        F_range_str = ui.gen_range_str(F_min, F_max)
        text += '    {:<16} {}'.format('Range:', F_range_str) + '\n'

        F_mean = self['F_mean'] if 'F_mean' in self else np.mean(self['F'].ravel())
        text += '    {:<16} {:<.3f}'.format('Mean:', F_mean) + '\n'

        F_var = self['F_var'] if 'F_var' in self else np.var(self['F'].ravel())
        text += '    {:<16} {:<.3f}'.format('Variance:', F_var) + '\n'

        text += '  {:<18} {}'.format('Fingerprint:', ui.unicode_str(self['md5']))

        idx = np.random.choice(n_mols, 1)[0]
        r = self['R'][idx, :, :]
        e = np.squeeze(self['E'][idx]) if 'E' in self else None
        f = self['F'][idx, :, :]
        lattice = self['lattice'] if 'lattice' in self else None

        text += '\n' + ui.white_bold_str('Example geometry') + ' (no. {:,}, chosen randomly)'.format(idx + 1) + '\n'
        xyz_info_str = 'Copy & paste the string below into Jmol (www.jmol.org), Avogadro (www.avogadro.cc), etc. to visualize one of the geometries from this dataset. A new example will be drawn on each run.'
        xyz_info_str = ui.wrap_str(xyz_info_str, width=MAX_PRINT_WIDTH - 2)
        xyz_info_str = ui.indent_str(xyz_info_str, 2)
        text += xyz_info_str + '\n' + '\n'

        xyz_str = io.generate_xyz_str(r, self['z'], e=e, f=f, lattice=lattice)
        xyz_str = ui.indent_str(xyz_str, 2)

        cut_str = '---- COPY HERE '
        cut_str_reps = int(np.floor((MAX_PRINT_WIDTH - 6) / len(cut_str)))
        cutline_str = ui.gray_str('  -' + cut_str * cut_str_reps + '-----')

        text += cutline_str + '\n'
        text += xyz_str + '\n'
        text += cutline_str + '\n'
        return text


class Model:
    """
        Model object

        This object serve as recipes for model creation,
        summarizing the configuration of one particular training run.
        Training and test points are sampled from the provided dataset,
        without replacement. If the same dataset if given for training
        and testing, the subsets are drawn without overlap.

        Each model also contains a choice for the hyper-parameters of the
        training process and the MD5 fingerprints of the used datasets.
        """
    def __init__(self, trainer=None, max_processes=None, use_torch=False,
                 dataset=None, n_train=0, save_dataset=False,
                 valid_dataset=None, n_valid=0, save_valid_dataset=False,
                 model=None, callback=dummy_callback,
                 solver=None,
                 **kwargs):
        """
            Constructor for the Model object.
            The resulting Task object is pre-trained and pre-tested.

            Parameters
            ----------
                trainer : :obj:`sgdml.train.GDMLTrain`
                    predefined GDMLTrain object. overwrite max_processes and use_torch Parameters
                dataset : :obj:'sgdml.core.Dataset'
                    Data structure of custom type :obj:`dataset` containing
                    training dataset.
                n_train : int
                    number of trainig point to sample from the training dataset.
                save_dataset: bool, optional
                    True: save the dataset object into the model for quick
                    change of the n_train via the update_hyperparameters method.
                    False: not saving the training dataset into the Model object.
                valid_dataset : :obj:'sgdml.core.Dataset', optional
                    Data structure of custom type :obj:`dataset` containing
                    training dataset.
                n_valid : int, optional
                    number of validation point to sample from the validation dataset.
                save_valid_dataset: bool, optional
                    True: save the dataset object into the model for quick
                    change of the n_valid via the update_hyperparameters() method.
                    False: not saving the validation dataset into the Model object.
                model : :obj:'sgdml.core.Model', optional
                    A precreate Model object to use as template for a new Model object.
                    overwrite every other option.
                callback : :obj:'sgdml.utils.ui.callback', optional
                    A callback object for monitoring the training and testing processes.
                kwargs :
                    Parameters for the creation of :obj: 'sgdml.core.Task'
                    object as a holder for the model hyperparameters
            Returns
            -------
                sgdml.core.Model
                    pre-trained and pre-tested type :obj:`sgdml.core.Model`.

            Raises
            ------
                ValueError
                    If a reconstruction of the potential energy surface is requested,
                    but the energy labels are missing in the dataset.
            """

        self.max_processes = max_processes
        self.use_torch = use_torch
        self.trainer = trainer

        if model is not None:
            for k, v in model.items():
                self[k] = v
            return
        if isinstance(trainer, Model):
            for k, v in trainer.items():
                self[k] = v
            return

        if 'task' not in kwargs.keys():
            self.task = Task(dataset, n_train,
                             valid_dataset=valid_dataset, n_valid=n_valid,
                             callback=callback, max_processes = self.max_processes,
                             solver=solver,
                             **kwargs)
        else:
            self.task = kwargs['task']
        for k, v in self.task.items():
            self[k] = v
        self.callback = callback
        self.dataset = None
        self.valid_dataset = None
        if save_dataset:
            self.dataset = dataset
        if save_valid_dataset:
            self.valid_dataset = valid_dataset
        self.solver = solver

    def update_hyperparameters(self, train_dataset=None, n_train=None,
                               valid_dataset=None, n_valid=None,
                               **kwargs):
        """
        Update the Task hyperparameters.

        Parameters
        ----------
            train_dataset :obj:`sgdml.core.Dataset`
                Data structure of custom type :obj:`dataset` containing a new
                train dataset
            n_train: int
                update value for number of training point for the model
            valid_dataset :obj:`sgdml.core.Dataset`
                Data structure of custom type :obj:`dataset` containing a new
                validation dataset
            n_valid: int
                update value for number of validation point for the model

                """

        if n_train is not None:
            if train_dataset is None and self.dataset is not None:
                train_dataset = self.dataset
            kwargs['n_train'] = n_train
        if 'idxs_train' in kwargs.keys():
            if train_dataset is None and self.dataset is not None:
                train_dataset = self.dataset
        if n_valid is not None:
            if valid_dataset is None and self.valid_dataset is not None:
                valid_dataset = self.valid_dataset
            kwargs['n_valid'] = n_valid
        self.task.update_hyperparameters(train_dataset=train_dataset, valid_dataset=valid_dataset,
                                         **kwargs)
        for k, v in self.task.items():
            self[k] = v

        self['type'] = 'm'

    def train(self, callback=None):
        del_trainer = False
        callback = callback if callback is not None else self.callback
        if self.trainer is None:
            self.trainer = GDMLTrain(self.max_processes, self.use_torch)
            del_trainer = True
        self.task.solver_name = self.solver if self.solver is not None else 'analytic'
        for k, v in self.task.items():
            self.task[k] = self[k]
        self.model_dict = self.trainer.train(self.task, callback=callback)
        for k, v in self.model_dict.items():
            self[k] = v
        if del_trainer:
            del self.trainer
            self.trainer = None
            global glob

            if 'glob' in globals():
                del glob

        self.predictor = None


    def prepare_parallel(self, n_bulk=1, n_reps=1):

        """
        run the prepare_parallel predictor :obj: 'sgdml.predict.GDMLpredict' method
        to find and set the optimal parallelization parameters for the
        currently loaded model, running on a particular system. The result
        also depends on the number of geometries `n_bulk` that will be
        passed at once when calling the `predict` function.

        This function runs a benchmark in which the prediction routine is
        repeatedly called `n_reps`-times (default: 1) with varying parameter
        configurations, while the runtime is measured for each one. The
        optimal parameters are then cached for fast retrival in future
        calls of this function.

        Parameters
        ----------
                n_bulk : int, optional
                        Number of geometries that will be passed to the
                        `predict` function in each call (performance
                        will be optimized for that exact use case).
                n_reps : int, optional
                        Number of repetitions (bigger value: more
                        accurate, but also slower).
        """
        if self.predictor is None:
            self.predictor = GDMLPredict(self,
                                         max_processes=self.max_processes, use_torch=self.use_torch)
        if not self.use_torch:
            self.callback(NOT_DONE, disp_str='Optimizing parallelism')

            gps, is_from_cache = self.predictor.prepare_parallel(n_bulk=n_bulk,n_reps=n_reps, return_is_from_cache=True)
            num_workers, batch_size, bulk_mp = (self.predictor.num_workers,
                                                self.predictor.chunk_size,
                                                self.predictor.bulk_mp,
                                                )

            self.callback(DONE,
                          disp_str='Optimizing parallelism'
                                   + (' (from cache)' if is_from_cache else ''),
                          sec_disp_str='%d workers %s/ chunks of %d'
                                       % (num_workers, '[MP] ' if bulk_mp else '', batch_size),
                          )
        else:
            print('Skipping multi-CPU benchmark, since torch is enabled.')# TODO: clarity!

    def predict(self, R, R_desc=None, R_d_desc=None):
        """
        Predict energy and forces for multiple geometries. This function
        can run on the GPU, if the optional PyTorch dependency is
        installed and `use_torch=True` was speciefied during
        initialization of this class.

        Optionally, the descriptors and descriptor Jacobians for the
        same geometries can be provided, if already available from some
        previous calculations.

        Note
        ----
                The order of the atoms in `R` is not arbitrary and must
                be the same as used for training the model.

        Parameters
        ----------
                R : :obj:`numpy.ndarray`
                        An 2D array of size M x 3N containing the
                        Cartesian coordinates of each atom of M
                        molecules.
                R_desc : :obj:`numpy.ndarray`, optional
                        An 2D array of size M x D containing the
                        descriptors of dimension D for M
                        molecules.
                R_d_desc : :obj:`numpy.ndarray`, optional
                        A 2D array of size M x D x 3N containing of the
                        descriptor Jacobians for M molecules. The descriptor
                        has dimension D with 3N partial derivatives with
                        respect to the 3N Cartesian coordinates of each atom.

        Returns
        -------
                :obj:`numpy.ndarray`
                        Energies stored in an 1D array of size M.
                :obj:`numpy.ndarray`
                        Forces stored in an 2D arry of size M x 3N.
        """
        if self.predictor is None:
            self.predictor = GDMLPredict(self,
                                         max_processes=self.max_processes, use_torch=self.use_torch)
        return self.predictor.predict(R, R_desc=None, R_d_desc=None)

    def test(self, dataset=None, n_test=None, is_test=False, overwrite=False):

        """
        test or validated the model on a dataset

        Note
        ----
        this function runs a validation if: is_test is False

        Parameters
        ----------
                dataset : :obj:`sgdml.core.Dataset`, optional
                        dataset to test the model on. if non is given the model will test on the saved validation dataset if present
                n_test : int, optional
                        number of datapoint to test the model on.
                        if None is given: validation: for validation n_valid will be used if present
                                                        else will use the full validation dataset will be used
                                            test: for test the full validation\test dataset will be used
                is_test : bool, optional
                        True: preform a test of the model.
                        False: preform a validation of the model
                overwrite : bool, optional
                        if to overwrite previous test/validation
        """

        if dataset is None and self.valid_dataset is not None:
            dataset = self.valid_dataset
            if is_test:
                print('Running test on validation dataset')
        # NOTE: this function runs a validation if:
        # is_test is False
        is_validation = not is_test
        if n_test is None:
            if is_validation:
                try:
                    n_test = 0 if self.n_valid is None else self.n_valid
                except AttributeError:
                    n_test = 0
            if is_test:
                n_test = 0

        F_rmse = []

        if not np.array_equal(self.z, dataset.z):
            raise AssistantError('Atom composition or order in dataset does not match the one in model.')

        if ('lattice' in self) is not ('lattice' in dataset):
            if 'lattice' in self:
                raise AssistantError('Model contains lattice vectors, but dataset does not.')
            elif 'lattice' in dataset:
                raise AssistantError('Dataset contains lattice vectors, but model does not.')

        if self.use_E:
            e_err = self.e_err
        f_err = self.f_err
        is_model_validated = not (np.isnan(np.array(f_err['mae'])) or np.isnan(np.array(f_err['rmse'])))

        if dataset.md5 != self.md5_valid:
            raise AssistantError('Fingerprint of provided validation dataset does not match the one in model file.')

        test_idxs = self.idxs_valid
        if is_test:
            # exclude training and/or test sets from validation set if necessary
            excl_idxs = np.empty((0,), dtype=np.uint)
            if dataset.md5 == self.md5_train:
                excl_idxs = np.concatenate([excl_idxs, self.idxs_train]).astype(np.uint)
            if dataset.md5 == self.md5_valid:
                excl_idxs = np.concatenate([excl_idxs, self.idxs_valid]).astype(np.uint)

            n_data = dataset.F.shape[0]
            n_data_eff = n_data - len(excl_idxs)

            if (n_test == 0 and n_data_eff != 0):  # test on all data points that have not been used for training or testing
                n_test = n_data_eff
                print('Test set size was automatically set to {:,} points.'.format(n_test))

            if n_test == 0 or n_data_eff == 0:
                print('Skipping! No unused points for test in provided dataset.')
                return
            elif n_data_eff < n_test:
                n_test = n_data_eff
                print('Test size reduced to {:d}. Not enough unused points in provided dataset.'.format(n_test))

            if 'E' in dataset:
                test_idxs = draw_strat_sample(dataset['E'], n_test, excl_idxs=excl_idxs)
            else:
                test_idxs = np.delete(np.arange(n_data), excl_idxs)
                print('Test dataset will be sampled with no guidance from energy labels (randomly)!\n'
                      + 'Note: Larger test datasets are recommended due to slower convergence of the error.')
        # shuffle to improve convergence of online error
        np.random.shuffle(test_idxs)

        # NEW

        z = dataset.z
        R = dataset.R[test_idxs, :, :]
        F = dataset.F[test_idxs, :, :]

        if self.use_E:
            E = dataset.E[test_idxs]

        b_size = min(1000, len(test_idxs))

        if self.use_torch:
            self.prepare_parallel(b_size)

        n_atoms = z.shape[0]

        if self.use_E:
            e_mae_sum, e_rmse_sum = 0, 0
        f_mae_sum, f_rmse_sum = 0, 0
        cos_mae_sum, cos_rmse_sum = 0, 0
        mag_mae_sum, mag_rmse_sum = 0, 0

        n_done = 0
        t = time.time()
        for i, b_range in enumerate(_batch(list(range(len(test_idxs))), b_size)):

            n_done_step = len(b_range)
            n_done += n_done_step

            r = R[b_range].reshape(n_done_step, -1)
            e_pred, f_pred = self.predict(r)

            # energy error
            if self.use_E:
                e = E[b_range]
                e_mae, e_mae_sum, e_rmse, e_rmse_sum = _online_err(np.squeeze(e) - e_pred, 1, n_done, e_mae_sum,
                                                                   e_rmse_sum)

            # force component error
            f = F[b_range].reshape(n_done_step, -1)
            f_mae, f_mae_sum, f_rmse, f_rmse_sum = _online_err(f - f_pred, 3 * n_atoms, n_done, f_mae_sum, f_rmse_sum)

            # magnitude error
            f_pred_mags = np.linalg.norm(f_pred.reshape(-1, 3), axis=1)
            f_mags = np.linalg.norm(f.reshape(-1, 3), axis=1)
            mag_mae, mag_mae_sum, mag_rmse, mag_rmse_sum = \
                _online_err(f_pred_mags - f_mags, n_atoms, n_done, mag_mae_sum, mag_rmse_sum)

            # normalized cosine error
            f_pred_norm = f_pred.reshape(-1, 3) / f_pred_mags[:, None]
            f_norm = f.reshape(-1, 3) / f_mags[:, None]
            cos_err = np.arccos(np.einsum('ij,ij->i', f_pred_norm, f_norm)) / np.pi
            cos_mae, cos_mae_sum, cos_rmse, cos_rmse_sum = _online_err(cos_err, n_atoms, n_done, cos_mae_sum,
                                                                       cos_rmse_sum)

            sps = n_done / (time.time() - t)  # examples per second
            disp_str = 'energy %.3f/%.3f, ' % (e_mae, e_rmse) if self.use_E else ''
            disp_str += 'forces %.3f/%.3f' % (f_mae, f_rmse)
            disp_str = ('{} errors (MAE/RMSE): '.format('Test' if is_test else 'Validation')
                        + disp_str)
            sec_disp_str = '@ %.1f geo/s' % sps if b_range is not None else ''

            self.callback(n_done,
                          len(test_idxs),
                          disp_str=disp_str,
                          sec_disp_str=sec_disp_str,
                          newline_when_done=False,
                          )
            if not self.use_E:
                e_mae, e_rmse = 0, 0
            try:
                self.callback(e_mae=e_mae, e_rmse=e_rmse,
                              f_mae=f_mae, f_rmse=f_rmse,
                              mag_mae=mag_mae, mag_rmse=mag_rmse,
                              cos_mae=cos_mae, cos_rmse=cos_rmse, iter=i)
            except:
                pass

        if is_test:
            self.callback(DONE, disp_str='Testing on {:,} points'.format(n_test),
                          sec_disp_str=sec_disp_str,
                          )
        else:
            self.callback(DONE, disp_str=disp_str, sec_disp_str=sec_disp_str)

        if self.use_E:
            e_rmse_pct = (e_rmse / e_err['rmse'] - 1.0) * 100
        f_rmse_pct = (f_rmse / f_err['rmse'] - 1.0) * 100

        # if func_called_directly and n_models == 1:
        if is_test:
            print(ui.white_bold_str('\nTest errors (MAE/RMSE)'))

            r_unit = 'unknown unit'
            e_unit = 'unknown unit'
            f_unit = 'unknown unit'
            if 'r_unit' in dataset and 'e_unit' in dataset:
                r_unit = dataset['r_unit']
                e_unit = dataset['e_unit']
                f_unit = str(dataset['e_unit']) + '/' + str(dataset['r_unit'])

            format_str = '  {:<18} {:>.4f}/{:>.4f} [{}]'
            if self.use_E:
                ui.print_two_column_str(format_str.format('Energy:', e_mae, e_rmse, e_unit),
                                        'relative to expected: {:+.1f}%'.format(e_rmse_pct), )

            ui.print_two_column_str(format_str.format('Forces:', f_mae, f_rmse, f_unit),
                                    'relative to expected: {:+.1f}%'.format(f_rmse_pct), )

            print(format_str.format('  Magnitude:', mag_mae, mag_rmse, r_unit))
            print(format_str.format('  Angle:', cos_mae, cos_rmse, '0-1, lower is better'))

        model_needs_update = (overwrite
                              or (is_test and self.n_test < len(test_idxs))
                              or (is_validation and not is_model_validated)
                              )
        if model_needs_update:
            if is_validation and overwrite:
                self.n_test = 0  # flag the model as not tested

            if is_test:
                self.n_test = len(test_idxs)
                self.md5_test = dataset.md5

            if self.use_E:
                self.e_err = {'mae': e_mae, 'rmse': e_rmse, }

            self.f_err = {'mae': f_mae, 'rmse': f_rmse}

            if is_test and self.n_test > 0:
                print('Expected errors were updated in model file.')

        else:
            add_info_str = (
                'the same number of' if self.n_test == len(test_idxs) else 'only {:,}'.format(len(test_idxs)))
            print('This model has previously been tested on {:,} points, '
                  'which is why the errors for the current test run with {} points have '
                  'NOT been used to update the model file.\n'.format(self.n_test, add_info_str)
                  )
            F_rmse.append(f_rmse)

    def solve(self, **kwargs):

        del_trainer = False
        if self.trainer is None:
            self.trainer = GDMLTrain(self.max_processes, self.use_torch)
            del_trainer = True

        # TODO: change to if self.solver == None
        assert self.solver == 'analytic' or self.solver == 'cg' or not self.solver == None # or solver == 'fk'

        n_train, n_atoms = self.R_train.shape[:2]

        self.desc = desc.Desc(
            n_atoms,
            interact_cut_off=self.interact_cut_off,
            max_processes=self.max_processes,
        )

        n_perms = self.perms.shape[0]
        tril_perms = np.array([self.desc.perm(p) for p in self.perms])

        dim_i = 3 * n_atoms
        dim_d = self.desc.dim

        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        self.tril_perms_lin = (tril_perms + perm_offsets).flatten('F')

        # TODO: check if all atoms are in span of lattice vectors, otherwise suggest that
        # rows and columns might have been switched.
        lat_and_inv = None
        if 'lattice' in self:
            try:
                lat_and_inv = (self.lattice, np.linalg.inv(self.lattice))
            except np.linalg.LinAlgError:
                raise ValueError(  # TODO: Document me
                    'Provided dataset contains invalid lattice vectors (not invertible). Note: Only rank 3 lattice vector matrices are supported.'
                )

            # # TODO: check if all atoms are within unit cell
            # for r in task['R_train']:
            #     r_lat = lat_and_inv[1].dot(r.T)
            #     if not (r_lat >= 0).all():
            #         # raise ValueError( # TODO: Document me
            #         #    'Some atoms appear outside of the unit cell! Please check lattice vectors in dataset file.'
            #         # )
            #         pass

        R = self.R_train.reshape(n_train, -1)
        self.R_desc, self.R_d_desc = self.desc.from_R(
            R,
            lat_and_inv=lat_and_inv,
            callback=partial(self.callback, disp_str='Generating descriptors and their Jacobians'),
        )

        # Generate label vector.
        E_train_mean = None
        y = self.F_train.ravel().copy()
        if self.use_E and self.use_E_cstr:
            E_train = self.E_train.ravel().copy()
            E_train_mean = np.mean(E_train)

            y = np.hstack((y, -E_train + E_train_mean))
            # y = np.hstack((n*Ft, (1-n)*Et))
        y_std = np.std(y)
        y /= y_std

        n_train, dim_d = self.R_d_desc.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms

        # Compress kernel based on symmetries
        col_idxs = np.s_[:]
        if 'cprsn_keep_atoms_idxs' in self:
            cprsn_keep_idxs_lin = (np.arange(dim_i).reshape(n_atoms, -1)[self.cprsn_keep_atoms_idxs, :].ravel())

            col_idxs = (cprsn_keep_idxs_lin[:, None] + np.arange(n_train) * dim_i).T.ravel()

        if self.callback is not None:
            self.callback = partial(
                self.callback,
                disp_str='Assembling kernel matrix',
            )
        self.K = self.trainer._assemble_kernel_mat(
            self.R_desc,
            self.R_d_desc,
            self.tril_perms_lin,
            self.sig,
            self.desc,
            use_E_cstr=self.use_E_cstr,
            col_idxs=col_idxs,
            callback=self.callback,
        )
        start = timeit.default_timer()

        alphas = self.solver(np.copy(self.K), np.copy(y.copy()), callback=self.callback, **kwargs).copy()
        self.y = y
        self.std = y_std
        stop = timeit.default_timer()
        if self.callback is not None:
            dur_s = (stop - start) / 2
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            self.callback(DONE,
                     disp_str='Training on {:,} points'.format(self.n_train),
                     sec_disp_str=sec_disp_str,
                     )
        self.alphas_F = alphas
        if self.use_E_cstr:
            self.alphas_E = alphas[-n_train:]
            self.alphas_F = alphas[:-n_train]

            # Recover integration consta nt.
            # Note: if energy constraints are included in the kernel (via 'use_E_cstr'), do not
            # compute the integration constant, but simply set it to the mean of the training energies
            # (which was subtracted from the labels before training).
        # compatibility with original code:


        if 'cprsn_keep_atoms_idxs' in self:
            cprsn_keep_idxs = self.cprsn_keep_atoms_idxs

            R_d_desc_full = self.desc.d_desc_from_comp(self.R_d_desc).reshape(
                n_train, dim_d, n_atoms, 3
            )
            R_d_desc_full = R_d_desc_full[:, :, cprsn_keep_idxs, :].reshape(
                n_train, dim_d, -1
            )

            r_d_desc_alpha = np.einsum(
                'kji,ki->kj', R_d_desc_full, self.alphas_F.reshape(n_train, -1)
            )

        else:

            # TOOD: why not use 'd_desc_dot_vec'?

            i, j = np.tril_indices(n_atoms, k=-1)
            alphas_F_exp = self.alphas_F.reshape(-1, n_atoms, 3)

            r_d_desc_alpha = np.einsum(
                'kji,kji->kj', self.R_d_desc, alphas_F_exp[:, j, :] - alphas_F_exp[:, i, :]
            )

        self.R_d_desc_alpha = r_d_desc_alpha
        self.type = 'm'
        self.c = 0.0
        self.R_desc = self.R_desc.T
        if self.use_E:
            c = (self._recov_int_const()
                if E_train_mean is None
                else E_train_mean)
            if c is None:
                # Something does not seem right. Turn off energy predictions for this model, only output force predictions.
                self.use_E = False
            else:
                self.c = c

        if del_trainer:
            del self.trainer
            self.trainer = None


    def _recov_int_const(self
    ):  # TODO: document e_err_inconsist return
        """
        Estimate the integration constant for a force field model.

        The offset between the energies predicted for the original training
        data and the true energy labels is computed in the least square sense.
        Furthermore, common issues with the user-provided datasets are self
        diagnosed here.

        Parameters
        ----------
            R_desc : :obj:`numpy.ndarray`, optional
                    An 2D array of size M x D containing the
                    descriptors of dimension D for M
                    molecules.
            R_d_desc : :obj:`numpy.ndarray`, optional
                    A 2D array of size M x D x 3N containing of the
                    descriptor Jacobians for M molecules. The descriptor
                    has dimension D with 3N partial derivatives with
                    respect to the 3N Cartesian coordinates of each atom.

        Returns
        -------
            float
                Estimate for the integration constant.

        Raises
        ------
            ValueError
                If the sign of the force labels in the dataset from
                which the model emerged is switched (e.g. gradients
                instead of forces).
            ValueError
                If inconsistent/corrupted energy labels are detected
                in the provided dataset.
            ValueError
                If different scales in energy vs. force labels are
                detected in the provided dataset.
        """
        gdml = GDMLPredict(self, max_processes=self.max_processes)  # , use_torch=self._use_torch

        n_train = self.E_train.shape[0]
        R = self.R_train.reshape(n_train, -1)
        E_pred, _ = gdml.predict(R, R_desc=self.R_desc.T, R_d_desc=self.R_d_desc)
        E_ref = np.squeeze(self.E_train)

        e_fact = np.linalg.lstsq(
            np.column_stack((E_pred, np.ones(E_ref.shape))), E_ref, rcond=-1
        )[0][0]
        corrcoef = np.corrcoef(E_ref, E_pred)[0, 1]

        # import matplotlib.pyplot as plt
        # sidx = np.argsort(E_ref)
        # plt.plot(E_ref[sidx])
        # c = np.sum(E_ref - E_pred) / E_ref.shape[0]
        # plt.plot(E_pred[sidx]+c)
        # plt.show()
        # sys.exit()

        # import matplotlib.pyplot as plt
        # sidx = np.argsort(F_ref)
        # plt.plot(F_ref[sidx])
        # c = np.sum(F_ref - F_pred) / F_ref.shape[0]
        # plt.plot(F_pred[sidx],'--')
        # plt.show()
        # sys.exit()

        if np.sign(e_fact) == -1:
            print('The provided dataset contains gradients instead of force labels (flipped sign). Please correct!\n'
                + ui.color_str('Note:', bold=True)
                + 'Note: The energy prediction accuracy of the model will thus neither be validated nor tested in the following steps!'
            )
            return None

        if corrcoef < 0.95:
            print('Inconsistent energy labels detected!\n'
                + 'The predicted energies for the training data are only weakly correlated with the reference labels (correlation coefficient {:.2f}) which indicates that the issue is most likely NOT just a unit conversion error.\n\n'.format(
                    corrcoef
                )
                + ui.color_str('Troubleshooting tips:\n', bold=True)
                + ui.wrap_indent_str(
                    '(1) ',
                    'Verify the correct correspondence between geometries and labels in the provided dataset.',
                )
                + '\n'
                + ui.wrap_indent_str(
                    '(2) ', 'Verify the consistency between energy and force labels.'
                )
                + '\n'
                + ui.wrap_indent_str('    - ', 'Correspondence correct?')
                + '\n'
                + ui.wrap_indent_str('    - ', 'Same level of theory?')
                + '\n'
                + ui.wrap_indent_str('    - ', 'Accuracy of forces?')
                + '\n'
                + ui.wrap_indent_str(
                    '(3) ',
                    'Is the training data spread too broadly (i.e. weakly sampled transitions between example clusters)?',
                )
                + '\n'
                + ui.wrap_indent_str(
                    '(4) ', 'Are there duplicate geometries in the training data?'
                )
                + '\n'
                + ui.wrap_indent_str(
                    '(5) ', 'Are there any corrupted data points (e.g. parsing errors)?'
                )
                + '\n\n'
                + ui.color_str('Note:', bold=True)
                + ' The energy prediction accuracy of the model will thus neither be validated nor tested in the following steps!'
            )
            return None

        if np.abs(e_fact - 1) > 1e-1:
            print('Different scales in energy vs. force labels detected!\n'
                + 'The integrated forces differ from the energy labels by factor ~{:.2f}, meaning that the trained model will likely fail to predict energies accurately.\n\n'.format(
                    e_fact
                )
                + ui.color_str('Troubleshooting tips:\n', bold=True)
                + ui.wrap_indent_str(
                    '(1) ', 'Verify consistency of units in energy and force labels.'
                )
                + '\n'
                + ui.wrap_indent_str(
                    '(2) ',
                    'Is the training data spread too broadly (i.e. weakly sampled transitions between example clusters)?',
                )
                + '\n\n'
                + ui.color_str('Note:', bold=True)
                + ' The energy prediction accuracy of the model will thus neither be validated nor tested in the following steps!'
            )
            return None

        # Least squares estimate for integration constant.
        return np.sum(E_ref - E_pred) / E_ref.shape[0]

    def to_dict(self):
        model = {
            'type': 'm',
            'code_version': __version__,
            'dataset_name': self['dataset_name'],
            'dataset_theory': self['dataset_theory'],
            'solver_name': self.solver,
            'solver_tol': self['solver_tol'],
            'norm_y_train': self.norm_y_train,
            'n_inducing_pts_init': self.n_inducing_pts_init if self.solver == 'cg' else None,
            'z': self['z'],
            'idxs_train': self['idxs_train'],
            'md5_train': self['md5_train'],
            'idxs_valid': self['idxs_valid'],
            'md5_valid': self['md5_valid'],
            'n_test': 0,
            'md5_test': None,
            'f_err': self.f_err,

            'R_desc': self.R_desc,
            'R_d_desc_alpha': self.R_d_desc_alpha,
            'interact_cut_off': self['interact_cut_off'],
            'c': self.c,
            'std': self.std,
            'sig': self['sig'],
            'lam': self['lam'],
            'alphas_F': self.alphas_F,
            'perms': self['perms'],
            'tril_perms_lin': self.tril_perms_lin,
            'use_E': self['use_E'],
            'use_cprsn': self['use_cprsn'],
        }

        if 'solver_resid' in self.keys():
            model['solver_resid'] = self.solver_resid  # residual of solution (cg solver)

        if 'solver_iters'  in self.keys():
            model[
                'solver_iters'
            ] = self.solver_iters  # number of iterations performed to obtain solution (cg solver)

        if 'inducing_pts_idxs' in self.keys():
            model['inducing_pts_idxs'] = self.inducing_pts_idxs

        if self['use_E']:
            model['e_err'] = self.e_err,
            if self['use_E_cstr']:
                model['alphas_E'] = self.alphas_E

        if 'lattice' in self.keys():
            model['lattice'] = self['lattice']

        if 'r_unit' in self.keys() and 'e_unit' in self.keys():
            model['r_unit'] = self['r_unit']
            model['e_unit'] = self['e_unit']
        return  model

    def __str__(self):
        text = ui.white_bold_str('Model properties') + '\n'

        text += '  {:<18} {} \n'.format('Dataset:', self.dataset_name)

        n_atoms = len(self.z)
        text += '  {:<18} {:<d} \n'.format('Atoms:', n_atoms)

        lat_str = 'n/a'
        if self['lattice'] is not None:
            lat_str = ui.gen_lattice_str(self['lattice'])
            lengths, angles = io.lattice_vec_to_par(self['lattice'])

        text += '  {:<18} {} '.format('Lattice:', lat_str) + '\n'
        if self['lattice'] is not None:
            text += '    {:<16} a = {:g}, b = {:g}, c = {:g}'.format('Lengths:', *lengths) + '\n'
            text += '    {:<16} alpha = {:g}, beta = {:g}, gamma = {:g}'.format('Angles [deg]:', *angles) + '\n'

        text += '  {:<18} {:<d}'.format('Symmetries:', len(self.perms)) + '\n'

        _, cprsn_keep_idxs = np.unique(np.sort(self.perms, axis=0), axis=1, return_index=True)
        n_atoms_kept = cprsn_keep_idxs.shape[0]
        text += '  {:<18} {}'.format(
            'Compression:',
            '{:<d} effective atoms'.format(n_atoms_kept)
            if 'use_cprsn' in self and self.use_cprsn
            else 'n/a',
        ) + '\n'

        text += '  {:<18}'.format('Hyper-parameters:', len(self.perms)) + '\n'
        text += '    {:<16} {:<d}'.format('Length scale:', self.sig) + '\n'

        if 'lam' in self:
            text += '    {:<16} {:<.0e}'.format('Regularization:', self.lam) + '\n'

        n_train = len(self.idxs_train)

        text += '  {:<18} {:,} points'.format('Trained on:', n_train) + ' from \'' \
                + ui.unicode_str(self.md5_train) + '\'' + '\n'

        if self.use_E:
            e_err = self.e_err
        f_err = self.f_err

        n_valid = len(self.idxs_valid)
        is_valid = not f_err['mae'] is None and not f_err['rmse'] is None
        text += '  {:<18} {}{:,} points'.format('Validated on:', '' if is_valid else '[pending] ', n_valid) + \
                ' from \'' + ui.unicode_str(self.md5_valid) + '\'' + '\n'
        if 'n_test' in self.keys():
            if self.n_test != 0:
                n_test = int(self.n_test)
                is_test = True
            else:
                is_test = False
        else:
            is_test = False
        if is_test:
            n_test = int(self.n_test)
            ui.print_two_column_str(
                '  {:<18} {:,} points'.format('Tested on:', n_test),
                'from \'' + self.md5_test + '\'',
            )
        else:
            text += '  {:<18} {}'.format('Test:', '[pending]') + '\n'

        e_unit = 'unknown unit'
        f_unit = 'unknown unit'
        if 'r_unit' in self and 'e_unit' in self:
            e_unit = self.e_unit
            f_unit = str(self.e_unit) + '/' + str(self.r_unit)

        if is_valid:
            action_str = 'Validation' if not is_valid else 'Expected test'
            text += '  {:<18}'.format('{} errors (MAE/RMSE):'.format(action_str)) + '\n'
            if self.use_E:
                text += '    {:<16} {:>.4f}/{:>.4f} [{}]'.format(
                    'Energy:', e_err['mae'], e_err['rmse'], e_unit
                ) + '\n'

            text += '    {:<16} {:>.4f}/{:>.4f} [{}]'.format(
                'Forces:', f_err['mae'], f_err['rmse'], f_unit
            ) + '\n'
        return text

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)


class Task:

    """
    Task object

    This object serve as recipes for model creation,
    summarizing the configuration of one particular training run.
    Training and test points are sampled from the provided dataset,
    without replacement. If the same dataset if given for training
    and testing, the subsets are drawn without overlap.

    Each task also contains a choice for the hyper-parameters of the
    training process and the MD5 fingerprints of the used datasets.
    """
    def __init__(self, train_dataset,
                 n_train,
                 valid_dataset=None,
                 n_valid=None,
                 sig=1,
                 lam=1e-15,
                 use_sym=True,
                 use_E=True,
                 use_E_cstr=False,
                 use_cprsn=False,
                 max_processes=None,
                 model0=None,  # TODO: document me
                 solver='analytic',  # TODO: document me
                 solver_tol=1e-4,  # TODO: document me
                 n_inducing_pts_init=None,  # TODO: document me
                 interact_cut_off=None,  # TODO: document me
                 callback=None, ):  # TODO: document me

        """
                Constructor for the Task object.
                The resulting Task object is pre-trained and pre-tested.

                Parameters
                ----------
                    train_dataset : :obj:`sgdml.core.Dataset`
                        Data structure of custom type :obj:`dataset` containing
                        train dataset.
                    n_train : int
                        Number of training points to sample.
                    valid_dataset : :obj:`sgdml.core.Dataset`, optional
                        Data structure of custom type :obj:`dataset` containing
                        validation dataset.
                    n_valid : int, optional
                        Number of validation points to sample.
                    sig : int
                        Hyper-parameter (kernel length scale).
                    lam : float, optional
                        Hyper-parameter lambda (regularization strength).
                    use_sym : bool, optional
                        True: include symmetries (sGDML), False: GDML.
                    use_E : bool, optional
                        True: reconstruct force field with corresponding potential energy surface,
                        False: ignore energy during training, even if energy labels are available
                               in the dataset. The trained model will still be able to predict
                               energies up to an unknown integration constant. Note, that the
                               energy predictions accuracy will be untested.
                    use_E_cstr : bool, optional
                        True: include energy constraints in the kernel,
                        False: default (s)GDML.
                    use_cprsn : bool, optional
                        True: compress kernel matrix along symmetric degrees of
                        freedom,
                        False: train using full kernel matrix

                Returns
                -------
                    sgdml.core.Task
                        pre-trained and pre-tested type :obj:`sgdml.core.Task`.

                Raises
                ------
                    ValueError
                        If a reconstruction of the potential energy surface is requested,
                        but the energy labels are missing in the dataset.
                """

        if use_E and 'E' not in train_dataset:
            raise ValueError(
                'No energy labels found in dataset!\n'
                + 'By default, force fields are always reconstructed including the\n'
                + 'corresponding potential energy surface (this can be turned off).\n'
                + 'However, the energy labels are missing in the provided dataset.\n'
            )

        self.type = 't'
        self.code_version = sgdml.__version__
        self.dataset_name = train_dataset.name
        self.dataset_theory = train_dataset.theory
        self.z = train_dataset.z
        self.sig = sig
        self.lam = lam
        self.use_E = use_E
        self.use_sym = use_sym
        self.use_cprsn = use_cprsn
        self.max_processes = max_processes
        self.solver_name = solver if solver is not None else 'analytic'
        self.solver_tol = solver_tol
        self.n_inducing_pts_init = alias_int(n_inducing_pts_init) if n_inducing_pts_init is not None else None
        self.interact_cut_off = interact_cut_off
        self.model0 = model0
        self.callback = callback
        self.use_E = use_E
        self.n_train = n_train
        self.n_valid = n_valid

        self.idxs_train = None
        self.md5_train = None
        self.idxs_valid = None
        self.md5_valid = None
        if self.use_E:
            self.e_err = {'mae': np.NaN, 'rmse': np.NaN, }
        self.f_err = {'mae': np.NaN, 'rmse': np.NaN}
        self.R_train = None
        self.F_train = None

        if use_E and 'E' not in train_dataset:
            raise ValueError('No energy labels found in dataset!\n'
                             + 'By default, force fields are always reconstructed including the\n'
                             + 'corresponding potential energy surface (this can be turned off).\n'
                             + 'However, the energy labels are missing in the provided dataset.\n')
        self.use_E_cstr = use_E and use_E_cstr

        if self.callback is not None:
            self.callback = partial(self.callback, disp_str='Hashing dataset(s)')
            self.callback(NOT_DONE)

        if self.callback is not None:
            self.callback(DONE)

        if self.model0 is not None and (train_dataset.md5 != self.model0['md5_train']
                                        or valid_dataset.md5 != self.model0['md5_valid']):
            raise ValueError('Provided training and/or validation dataset(s) do(es) not match the ones in the initial '
                             'model.')

        self.update_model_index(train_dataset, n_train)
        if valid_dataset is not None:
            self.update_model_valid_index(n_valid, valid_dataset)
        if use_E:
            self.E_train = train_dataset.E[self.idxs_train]

        lat_and_inv = None
        if 'lattice' in train_dataset:
            self.lattice = train_dataset.lattice

            try:
                lat_and_inv = (self.lattice, np.linalg.inv(self.lattice))
            except np.linalg.LinAlgError:
                raise ValueError(  # TODO: Document me
                    'Provided dataset contains invalid lattice vectors (not invertible). Note: Only rank 3 lattice '
                    'vector matrices are supported.')

        if 'r_unit' in train_dataset and 'e_unit' in train_dataset:
            self.r_unit = train_dataset.r_unit
            self.e_unit = train_dataset.e_unit

    def update_model_index(self, train_dataset,n_train=None, idxs_train=None):

        """
        Update the Task train indices while taking into account indices of the base model0
        and permutations in the training dataset

        Parameters
        ----------
            train_dataset : :obj:`sgdml.core.Dataset`
                Data structure of custom type :obj:`dataset` containing
                train dataset.
            n_train : int
                Number of training points to sample.

        """

        md5_train = io.dataset_md5(train_dataset)
        m0_excl_idxs = np.array([], dtype=np.uint)
        m0_n_train, m0_n_valid = 0, 0
        if self.model0 is not None:
            m0_idxs_train = self.model0['idxs_train']  # TODO: Change the dict syntax
            m0_idxs_valid = self.model0['idxs_valid']
            m0_n_train = m0_idxs_train.shape[0]
            self.m0_excl_idxs = np.concatenate((m0_idxs_train, m0_idxs_valid)).astype(np.uint)
        else:
            self.m0_excl_idxs = np.array([])
        # TODO: handle smaller training/validation set

        if self.callback is not None:
            callback = partial(self.callback, disp_str='Sampling training and validation subsets')
            callback(NOT_DONE)

        if idxs_train is None:
            if 'E' in train_dataset:
                idxs_train = draw_strat_sample(train_dataset.E, n_train - m0_n_train, m0_excl_idxs)
            else:
                idxs = np.arange(train_dataset.F.shape[0])
                not_use_idx = np.setdiff1d(idxs, self.m0_excl_idxs, assume_unique=True)  # DONE: m0 handling
                idxs_train = np.random.choice(not_use_idx, n_train - m0_n_train, replace=False)
            if self.model0 is not None:
                idxs_train = np.concatenate((m0_idxs_train, idxs_train)).astype(np.uint)

        if self.callback is not None:
            self.callback(DONE)

        self.idxs_train = idxs_train
        self.md5_train = md5_train
        self.R_train = train_dataset.R[self.idxs_train, :, :]
        self.F_train = train_dataset.F[self.idxs_train, :, :]

        if self.model0 is None:
            if self.use_sym:
                n_train = self.R_train.shape[0]
                R_train_sync_mat = self.R_train
                if n_train > 1000:
                    R_train_sync_mat = self.R_train[np.random.choice(n_train, 1000, replace=False), :, :]
                    print('Symmetry search has been restricted to a random subset of 1000/{:d} '
                          'training points for faster convergence.'.format(n_train))

                # TOOD: PBCs disabled when matching (for now).
                # task['perms'] = perm.find_perms(
                #    R_train_sync_mat, train_dataset['z'], lat_and_inv=lat_and_inv, max_processes=self._max_processes,
                # )
                self.perms = perm.find_perms(R_train_sync_mat,
                                             train_dataset['z'],
                                             lat_and_inv=None,
                                             callback=self.callback,
                                             max_processes=self.max_processes)

                # NEW

                USE_FRAG_PERMS = False

                if USE_FRAG_PERMS:
                    frag_perms = perm.find_frag_perms(R_train_sync_mat,
                                                      train_dataset['z'],
                                                      lat_and_inv=None,
                                                      max_processes=self.max_processes)
                    self.perms = np.vstack((self.perms, frag_perms))
                    self.perms = np.unique(self.perms, axis=0)

                    print('| Keeping ' + str(self.perms.shape[0]) + ' unique permutations.')

                # NEW

            else:
                self.perms = np.arange(train_dataset.R.shape[1])[None, :]  # no symmetries
        else:
            self.perms = self.model0['perms']  # TODO: change to obj syntax
            print('Reusing permutations from initial model.')

        if self.model0 is not None:

            n_train, n_atoms = self.R_train.shape[:2]

            if 'alphas_F' in self.model0:
                print('Reusing alphas from initial model.')

                # Pad existing alphas, if this training dataset is larger than the one in self.model0
                alphas0_F_padding = np.ones(((n_train - m0_n_train) * n_atoms * 3,)) * np.mean(
                    self.model0['alphas_F'])  # TODO: update to obj syntax
                self.alphas0_F = np.append(self.model0['alphas_F'], alphas0_F_padding)  # TODO: update to obj syntax

            if 'alphas_E' in self.model0:
                # Pad existing alphas, if this training dataset is larger than the one in model0
                alphas0_E_padding = np.ones(((n_train - m0_n_train) * n_atoms,)) * np.mean(self.model0['alphas_E'])
                self.alphas0_E = np.append(self.model0['alphas_E'], alphas0_E_padding)  # TODO: update to obj syntax
        # Which atoms can we keep, if we exclude all symmetric ones?
        n_perms = self.perms.shape[0]
        if self.use_cprsn and n_perms > 1:
            _, cprsn_keep_idxs = np.unique(np.sort(self.perms, axis=0), axis=1, return_index=True)
            self.cprsn_keep_atoms_idxs = cprsn_keep_idxs

    def update_model_valid_index(self, n_valid, valid_dataset):

        """
        Update the Task validation indices while taking into account indices of the base model0.
        If using the same dataset for the training and validation will omit training indices.

        Parameters
        ----------
            n_valid : int
                Number of training points to sample.
            valid_dataset : :obj:`sgdml.core.Dataset`
                Data structure of custom type :obj:`dataset` containing
                train dataset.

        """

        md5_valid = io.dataset_md5(valid_dataset)
        if self.model0 is not None:
            m0_idxs_valid = self.model0['idxs_valid']  # TODO: Change the dict syntax
            m0_n_valid = m0_idxs_valid.shape[0]
        else:
            m0_n_valid = 0

        excl_idxs = (self.idxs_train if self.md5_train == self.md5_valid
                     else np.array([], dtype=np.uint))  # TODO: TEST CASE: differnt test and val sets and m0
        excl_idxs = np.concatenate((self.m0_excl_idxs, excl_idxs)).astype(np.uint)
        if 'E' in valid_dataset:
            idxs_valid = draw_strat_sample(valid_dataset['E'], n_valid - m0_n_valid, excl_idxs)
        else:
            idxs_valid_all = np.setdiff1d(np.arange(valid_dataset['F'].shape[0]), excl_idxs, assume_unique=True)
            idxs_valid = np.random.choice(idxs_valid_all, n_valid - m0_n_valid, replace=False)
            # TODO: m0 handling, zero handling
        self.idxs_valid = idxs_valid
        self.md5_valid = md5_valid

    def update_hyperparameters(self, train_dataset=None, valid_dataset=None, **kwargs):
        """
        Update the Task hyperparameters.

        Parameters
        ----------
            train_dataset :obj:`sgdml.core.Dataset`
                Data structure of custom type :obj:`dataset` containing a new
                train dataset
            valid_dataset :obj:`sgdml.core.Dataset`
                Data structure of custom type :obj:`dataset` containing a new
                validation dataset

        """
        for k, v in kwargs.items():
            self[k] = v
        if 'n_inducing_pts_init' in kwargs.keys():
            self['n_inducing_pts_init'] = alias_int(kwargs['n_inducing_pts_init'])
        if train_dataset is not None:
            idxs_train = kwargs['idxs_train'] if 'idxs_train' in kwargs.keys() else None
            self.update_model_index(train_dataset, self.n_train, idxs_train)
            if self.use_E:
                self.E_train = train_dataset.E[self.idxs_train]

        if valid_dataset is not None:
            self.update_model_valid_index(self.n_valid, valid_dataset)

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)


class alias_str(str):
    def __init__(self, strg):
        super().__init__()
        self.str = strg

    def astype(self, astype):
        return astype(self.str)

    def __str__(self):
        return self.str


class alias_int(int):
    def __init__(self, num):
        super().__init__()
        self.num = num

    def astype(self, astype):
        return astype(self.num)

    def copy(self):
        return self.num


def draw_strat_sample(T, n, excl_idxs=None):
    """
    Draw sample from dataset that preserves its original distribution.

    The distribution is estimated from a histogram were the bin size is
    determined using the Freedman-Diaconis rule. This rule is designed to
    minimize the difference between the area under the empirical
    probability distribution and the area under the theoretical
    probability distribution. A reduced histogram is then constructed by
    sampling uniformly in each bin. It is intended to populate all bins
    with at least one sample in the reduced histogram, even for small
    training sizes.

    Parameters
    ----------
        T : :obj:`numpy.ndarray`
            Dataset to sample from.
        n : int
            Number of examples.
        excl_idxs : :obj:`numpy.ndarray`, optional
            Array of indices to exclude from sample.

    Returns
    -------
        :obj:`numpy.ndarray`
            Array of indices that form the sample.
    """
    if excl_idxs is not None:
        if len(excl_idxs) == 0:
            excl_idxs = None

    if n == 0:
        return np.array([], dtype=np.uint)

    if T.size == n:  # TODO: this only works if excl_idxs=None
        assert excl_idxs is None
        return np.arange(n)

    if n == 1:
        idxs_all_non_excl = np.setdiff1d(
            np.arange(T.size), excl_idxs, assume_unique=True
        )
        return np.array([np.random.choice(idxs_all_non_excl)])

    # Freedman-Diaconis rule
    h = 2 * np.subtract(*np.percentile(T, [75, 25])) / np.cbrt(n)
    n_bins = int(np.ceil((np.max(T) - np.min(T)) / h)) if h > 0 else 1
    n_bins = min(
        n_bins, int(n / 2)
    )  # Limit number of bins to half of requested subset size.

    bins = np.linspace(np.min(T), np.max(T), n_bins, endpoint=False)
    idxs = np.digitize(T, bins)

    # Exclude restricted indices.
    if excl_idxs is not None and excl_idxs.size > 0:
        idxs[excl_idxs] = n_bins + 1  # Impossible bin.

    uniq_all, cnts_all = np.unique(idxs, return_counts=True)

    # Remove restricted bin.
    if excl_idxs is not None and excl_idxs.size > 0:
        excl_bin_idx = np.where(uniq_all == n_bins + 1)
        cnts_all = np.delete(cnts_all, excl_bin_idx)
        uniq_all = np.delete(uniq_all, excl_bin_idx)

    # Compute reduced bin counts.
    reduced_cnts = np.ceil(cnts_all / np.sum(cnts_all, dtype=float) * n).astype(int)
    reduced_cnts = np.minimum(
        reduced_cnts, cnts_all
    )  # limit reduced_cnts to what is available in cnts_all

    # Reduce/increase bin counts to desired total number of points.
    reduced_cnts_delta = n - np.sum(reduced_cnts)

    while np.abs(reduced_cnts_delta) > 0:
        # How many members can we remove from an arbitrary bucket, without any bucket with more than one member going to zero?
        max_bin_reduction = np.min(reduced_cnts[np.where(reduced_cnts > 1)]) - 1

        # Generate additional bin members to fill up/drain bucket counts of subset. This array contains (repeated) bucket IDs.
        outstanding = np.random.choice(
            uniq_all,
            min(max_bin_reduction, np.abs(reduced_cnts_delta)),
            p=(reduced_cnts - 1) / np.sum(reduced_cnts - 1, dtype=float),
            replace=True,
        )
        uniq_outstanding, cnts_outstanding = np.unique(
            outstanding, return_counts=True
        )  # Aggregate bucket IDs.

        outstanding_bucket_idx = np.where(
            np.in1d(uniq_all, uniq_outstanding, assume_unique=True)
        )[
            0
        ]  # Bucket IDs to Idxs.
        reduced_cnts[outstanding_bucket_idx] += (
                np.sign(reduced_cnts_delta) * cnts_outstanding
        )
        reduced_cnts_delta = n - np.sum(reduced_cnts)

    # Draw examples for each bin.
    idxs_train = np.empty((0,), dtype=int)
    for uniq_idx, bin_cnt in zip(uniq_all, reduced_cnts):
        idx_in_bin_all = np.where(idxs.ravel() == uniq_idx)[0]
        idxs_train = np.append(
            idxs_train, np.random.choice(idx_in_bin_all, bin_cnt, replace=False)
        )

    return idxs_train


def select(validated_models, overwrite=False, max_processes=None, model_file=None, command=None,
           **kwargs):  # noqa: C901
    any_model_not_validated = False
    any_model_is_tested = False

    if len(validated_models) > 1:
        use_E = True

        rows = []
        data_names = ['sig', 'MAE', 'RMSE', 'MAE', 'RMSE']
        for i, model in enumerate(validated_models):
            use_E = model['use_E']

            if i == 0:
                idxs_train = set(model['idxs_train'])
                md5_train = model['md5_train']
                idxs_valid = set(model['idxs_valid'])
                md5_valid = model['md5_valid']
            else:

                if (md5_train != model['md5_train'] or md5_valid != model['md5_valid']
                        or idxs_train != set(model['idxs_train']) or idxs_valid != set(model['idxs_valid'])):
                    raise AssistantError(
                        '{} contains models trained or validated on different datasets.'.format(1))

            e_err = {'mae': 0.0, 'rmse': 0.0}
            try:
                if model['use_E']:
                    e_err = model['e_err'].item()
                f_err = model['f_err'].item()
            except:
                if model['use_E']:
                    e_err = model['e_err']
                f_err = model['f_err']

            is_model_validated = not (np.isnan(f_err['mae']) or np.isnan(f_err['rmse']))
            if not is_model_validated:
                any_model_not_validated = True

            is_model_tested = model['n_test'] > 0
            if is_model_tested:
                any_model_is_tested = True

            rows.append([model['sig'], e_err['mae'], e_err['rmse'], f_err['mae'], f_err['rmse']])

        if any_model_not_validated:
            print('One or more models in the given directory have not been validated yet.\n'
                  + 'This is required before selecting the best performer.')
            print()
            return

        if any_model_is_tested:
            print(
                'One or more models in the given directory have already been tested. This means that their recorded expected errors are test errors, not validation errors. However, one should never perform model selection based on the test error!\n'
                + 'Please run the validation command (again) with the overwrite option \'-o\', then this selection command.')
            return

        f_rmse_col = [row[4] for row in rows]
        best_idx = f_rmse_col.index(min(f_rmse_col))  # idx of row with lowest f_rmse
        best_sig = rows[best_idx][0]

        rows = sorted(rows, key=lambda col: col[0])  # sort according to sigma
        print(ui.white_bold_str('Cross-validation errors'))
        print(' ' * 7 + 'Energy' + ' ' * 6 + 'Forces')
        print((' {:>3} ' + '{:>5} ' * 4).format(*data_names))
        print(' ' + '-' * 27)
        format_str = ' {:>3} ' + '{:5.2f} ' * 4
        format_str_no_E = ' {:>3}     -     - ' + '{:5.2f} ' * 2
        for row in rows:
            if use_E:
                row_str = format_str.format(*row)
            else:
                row_str = format_str_no_E.format(*[row[0], row[3], row[4]])

            if row[0] != best_sig:
                row_str = ui.gray_str(row_str)
            print(row_str)
        print()

        sig_col = [row[0] for row in rows]
        if best_sig == min(sig_col) or best_sig == max(sig_col):
            print('The optimal sigma lies on the boundary of the search grid.\n'
                  + 'Model performance might improve if the search grid is extended in direction sigma {} {:d}.'.format(
                '<' if best_idx == 0 else '>', best_sig))

    else:  # only one model available
        print('Skipping model selection step as there is only one model to select.')

        best_idx = 0
    best_model = validated_models[best_idx]

    return best_model

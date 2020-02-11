#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2019 Stefan Chmiela
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

import argparse
import hashlib
import os
import re
import sys

import numpy as np

from . import ui

_z_str_to_z_dict = {
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
    'K': 19,
    'Ca': 20,
    'Sc': 21,
    'Ti': 22,
    'V': 23,
    'Cr': 24,
    'Mn': 25,
    'Fe': 26,
    'Co': 27,
    'Ni': 28,
    'Cu': 29,
    'Zn': 30,
    'Ga': 31,
    'Ge': 32,
    'As': 33,
    'Se': 34,
    'Br': 35,
    'Kr': 36,
    'Rb': 37,
    'Sr': 38,
    'Y': 39,
    'Zr': 40,
    'Nb': 41,
    'Mo': 42,
    'Tc': 43,
    'Ru': 44,
    'Rh': 45,
    'Pd': 46,
    'Ag': 47,
    'Cd': 48,
    'In': 49,
    'Sn': 50,
    'Sb': 51,
    'Te': 52,
    'I': 53,
    'Xe': 54,
    'Cs': 55,
    'Ba': 56,
    'La': 57,
    'Ce': 58,
    'Pr': 59,
    'Nd': 60,
    'Pm': 61,
    'Sm': 62,
    'Eu': 63,
    'Gd': 64,
    'Tb': 65,
    'Dy': 66,
    'Ho': 67,
    'Er': 68,
    'Tm': 69,
    'Yb': 70,
    'Lu': 71,
    'Hf': 72,
    'Ta': 73,
    'W': 74,
    'Re': 75,
    'Os': 76,
    'Ir': 77,
    'Pt': 78,
    'Au': 79,
    'Hg': 80,
    'Tl': 81,
    'Pb': 82,
    'Bi': 83,
    'Po': 84,
    'At': 85,
    'Rn': 86,
    'Fr': 87,
    'Ra': 88,
    'Ac': 89,
    'Th': 90,
    'Pa': 91,
    'U': 92,
    'Np': 93,
    'Pu': 94,
    'Am': 95,
    'Cm': 96,
    'Bk': 97,
    'Cf': 98,
    'Es': 99,
    'Fm': 100,
    'Md': 101,
    'No': 102,
    'Lr': 103,
    'Rf': 104,
    'Db': 105,
    'Sg': 106,
    'Bh': 107,
    'Hs': 108,
    'Mt': 109,
    'Ds': 110,
    'Rg': 111,
    'Cn': 112,
    'Uuq': 114,
    'Uuh': 116,
}
_z_to_z_str_dict = {v: k for k, v in _z_str_to_z_dict.items()}


def z_str_to_z(z_str):
    return np.array([_z_str_to_z_dict[x] for x in z_str])


def z_to_z_str(z):
    return [_z_to_z_str_dict[int(x)] for x in z]


def train_dir_name(
    dataset, n_train, use_sym, use_cprsn, use_E, use_E_cstr, model0=None
):

    theory_level_str = re.sub(r'[^\w\-_\.]', '.', str(dataset['theory']))
    theory_level_str = re.sub(r'\.\.', '.', theory_level_str)

    m0_str = 'm0-' if model0 is not None else ''
    sym_str = '-sym' if use_sym else ''
    cprsn_str = '-cprsn' if use_cprsn else ''
    noE_str = '-noE' if not use_E else ''
    Ecstr_str = '-Ecstr' if use_E_cstr else ''

    return '%ssgdml_cv_%s-%s-train%d%s%s%s%s' % (
        m0_str,
        dataset['name'].astype(str),
        theory_level_str,
        n_train,
        sym_str,
        cprsn_str,
        noE_str,
        Ecstr_str,
    )


def task_file_name(task):

    n_train = task['idxs_train'].shape[0]
    n_perms = task['perms'].shape[0]
    sig = np.squeeze(task['sig'])

    return 'task-train%d-sym%d-sig%04d.npz' % (n_train, n_perms, sig)


def model_file_name(task_or_model, is_extended=False):

    n_train = task_or_model['idxs_train'].shape[0]
    n_perms = task_or_model['perms'].shape[0]
    sig = np.squeeze(task_or_model['sig'])

    if is_extended:
        dataset = np.squeeze(task_or_model['dataset_name'])
        theory_level_str = re.sub(
            r'[^\w\-_\.]', '.', str(np.squeeze(task_or_model['dataset_theory']))
        )
        theory_level_str = re.sub(r'\.\.', '.', theory_level_str)
        return '%s-%s-train%d-sym%d.npz' % (dataset, theory_level_str, n_train, n_perms)
    return 'model-train%d-sym%d-sig%04d.npz' % (n_train, n_perms, sig)


def dataset_md5(dataset):

    md5_hash = hashlib.md5()

    keys = ['z', 'R']
    if 'E' in dataset:
        keys.append('E')
    keys.append('F')

    # only include new extra keys in fingerprint for 'modern' dataset files
    # 'code_version' was included from 0.4.0.dev1
    # opt_keys = ['lattice', 'e_unit', 'E_min', 'E_max', 'E_mean', 'E_var', 'f_unit', 'F_min', 'F_max', 'F_mean', 'F_var']
    # for k in opt_keys:
    #    if k in dataset:
    #        keys.append(k)

    for k in keys:
        d = dataset[k]
        if type(d) is np.ndarray:
            d = d.ravel()
        md5_hash.update(hashlib.md5(d).digest())

    return md5_hash.hexdigest().encode('utf-8')


# ## FILES

# Read geometry file (xyz format).
# R: (n_geo,3*n_atoms)
# z: (3*n_atoms,)
def read_xyz(file_path):

    with open(file_path, 'r') as f:
        n_atoms = None

        R, z = [], []
        for i, line in enumerate(f):
            line = line.strip()
            if not n_atoms:
                n_atoms = int(line)

            cols = line.split()
            file_i, line_i = divmod(i, n_atoms + 2)
            if line_i >= 2:
                R.append(list(map(float, cols[1:4])))
                if file_i == 0:  # first molecule
                    z.append(_z_str_to_z_dict[cols[0]])

        R = np.array(R).reshape(-1, 3 * n_atoms)
        z = np.array(z)

        f.close()
    return R, z


# Write geometry file (xyz format).
def write_geometry(filename, r, z, comment_str=''):

    r = np.squeeze(r)
    try:
        with open(filename, 'w') as f:
            f.write(str(len(r)) + '\n' + comment_str)
            for i, atom in enumerate(r):
                f.write('\n' + _z_to_z_str_dict[z[i]] + '\t')
                f.write('\t'.join(str(x) for x in atom))
    except IOError:
        sys.exit("ERROR: Writing xyz file failed.")


# Write geometry file (xyz format).
def generate_xyz_str(r, z, e=None, f=None, lattice=None):

    comment_str = ''
    if lattice is not None:
        comment_str += 'Lattice=\"{}\" '.format(
            ' '.join(['{:.12g}'.format(l) for l in lattice.T.ravel()])
        )
    if e is not None:
        comment_str += 'Energy={:.12g} '.format(e)
    comment_str += 'Properties=species:S:1:pos:R:3'
    if f is not None:
        comment_str += ':forces:R:3'

    species_str = '\n'.join([_z_to_z_str_dict[z_i] for z_i in z])
    r_f_str = ui.merge_col_str(ui.gen_mat_str(r)[0], ui.gen_mat_str(f)[0])

    xyz_str = str(len(r)) + '\n' + comment_str + '\n'
    xyz_str += ui.merge_col_str(species_str, r_f_str)

    return xyz_str


def lattice_vec_to_par(lat):

    cell = lat.T
    lengths = [np.linalg.norm(v) for v in cell]

    angles = []
    for i in range(3):
        j = i - 1
        k = i - 2

        ll = lengths[j] * lengths[k]
        if ll > 1e-16:
            x = np.dot(cell[j], cell[k]) / ll
            angle = 180.0 / np.pi * np.arccos(x)
        else:
            angle = 90.0
        angles.append(angle)

    return lengths, angles


### FILE HANDLING


def is_file_type(arg, type):
    """
    Validate file path and check if the file is of the specified type.

    Parameters
    ----------
        arg : :obj:`str`
            File path.
        type : {'dataset', 'task', 'model'}
            Possible file types.

    Returns
    -------
        (:obj:`str`, :obj:`dict`)
            Tuple of file path (as provided) and data stored in the
            file. The returned instance of NpzFile class must be
            closed to avoid leaking file descriptors.

    Raises
    ------
        ArgumentTypeError
            If the provided file path does not lead to a NpzFile.
        ArgumentTypeError
            If the file is not readable.
        ArgumentTypeError
            If the file is of wrong type.
        ArgumentTypeError
            If path/fingerprint is provided, but the path is not valid.
        ArgumentTypeError
            If fingerprint could not be resolved.
        ArgumentTypeError
            If multiple files with the same fingerprint exist.

    """

    # Replace MD5 dataset fingerprint with file name, if necessary.
    if type == 'dataset' and not arg.endswith('.npz') and not os.path.isdir(arg):
        dir = '.'
        if re.search(r'^[a-f0-9]{32}$', arg):  # arg looks similar to MD5 hash string
            md5_str = arg
        else:  # is it a path with a MD5 hash at the end?
            md5_str = os.path.basename(os.path.normpath(arg))
            dir = os.path.dirname(os.path.normpath(arg))

            if re.search(r'^[a-f0-9]{32}$', md5_str) and not os.path.isdir(
                dir
            ):  # path has MD5 hash string at the end, but directory is not valid
                raise argparse.ArgumentTypeError('{0} is not a directory'.format(dir))

        file_names = filter_file_type(dir, type, md5_match=md5_str)

        if not len(file_names):
            raise argparse.ArgumentTypeError(
                "No {0} files with fingerprint '{1}' found in '{2}'".format(
                    type, md5_str, dir
                )
            )
        elif len(file_names) > 1:
            error_str = "Multiple {0} files with fingerprint '{1}' found in '{2}'".format(
                type, md5_str, dir
            )
            for file_name in file_names:
                error_str += '\n       {0}'.format(file_name)

            raise argparse.ArgumentTypeError(error_str)
        else:
            arg = os.path.join(dir, file_names[0])

    if not arg.endswith('.npz'):
        argparse.ArgumentTypeError('{0} is not a .npz file'.format(arg))

    try:
        file = np.load(arg, allow_pickle=True)
    except Exception:
        raise argparse.ArgumentTypeError('{0} is not readable'.format(arg))

    if 'type' not in file or file['type'].astype(str) != type[0]:
        raise argparse.ArgumentTypeError('{0} is not a {1} file'.format(arg, type))

    return arg, file


def filter_file_type(dir, type, md5_match=None):
    """
    Filters all files from a directory that match a given type and (optionally)
    a given fingerprint.

    Parameters
    ----------
        arg : :obj:`str`
            File path.
        type : {'dataset', 'task', 'model'}
            Possible file types.
        md5_match : :obj:`str`, optional
            Fingerprint string.

    Returns
    -------
        :obj:`list` of :obj:`str`
            List of file names that match the specified type and fingerprint
            (if provided).

    Raises
    ------
        ArgumentTypeError
            If the directory contains unreadable .npz files.

    """

    file_names = []
    for file_name in sorted(os.listdir(dir)):
        if file_name.endswith('.npz'):
            file_path = os.path.join(dir, file_name)
            try:
                file = np.load(file_path, allow_pickle=True)
            except Exception:
                raise argparse.ArgumentTypeError(
                    '{0} contains unreadable .npz files'.format(arg)
                )

            if 'type' in file and file['type'].astype(str) == type[0]:

                if md5_match is None:
                    file_names.append(file_name)
                elif 'md5' in file and file['md5'] == md5_match:
                    file_names.append(file_name)

            file.close()

    return file_names


def is_valid_file_type(arg_in):
    """
    Check if file is either a valid dataset, task or model file.

    Parameters
    ----------
        arg_in : :obj:`str`
            File path.

    Returns
    -------
        (:obj:`str`, :obj:`dict`)
            Tuple of file path (as provided) and data stored in the
            file. The returned instance of NpzFile class must be
            closed to avoid leaking file descriptors.

    Raises
    ------
        ArgumentTypeError
            If the provided file path does not point to a supported
            file type.

    """

    arg, file = None, None
    try:
        arg, file = is_file_type(arg_in, 'dataset')
    except argparse.ArgumentTypeError:
        pass

    if file is None:
        try:
            arg, file = is_file_type(arg_in, 'task')
        except argparse.ArgumentTypeError:
            pass

    if file is None:
        try:
            arg, file = is_file_type(arg_in, 'model')
        except argparse.ArgumentTypeError:
            pass

    if file is None:
        raise argparse.ArgumentTypeError(
            '{0} is neither a dataset, task, nor model file'.format(arg)
        )

    return arg, file


def is_dir_with_file_type(arg, type, or_file=False):
    """
    Validate directory path and check if it contains files of the specified type.

    Note
    ----
        If a file path is provided, this function acts like its a directory with
        just one file.

    Parameters
    ----------
        arg : :obj:`str`
            File path.
        type : {'dataset', 'task', 'model'}
            Possible file types.
        or_file : bool
            If `arg` contains a file path, act like it's a directory
            with just a single file inside.

    Returns
    -------
        (:obj:`str`, :obj:`list` of :obj:`str`)
            Tuple of directory path (as provided) and a list of
            contained file names of the specified type.

    Raises
    ------
        ArgumentTypeError
            If the provided directory path does not lead to a directory.
        ArgumentTypeError
            If directory contains unreadable files.
        ArgumentTypeError
            If directory contains no files of the specified type.
    """

    if or_file and os.path.isfile(arg):  # arg: file path
        _, file = is_file_type(
            arg, type
        )  # raises exception if there is a problem with file
        file.close()
        file_name = os.path.basename(arg)
        file_dir = os.path.dirname(arg)
        return file_dir, [file_name]
    else:  # arg: dir

        if not os.path.isdir(arg):
            raise argparse.ArgumentTypeError('{0} is not a directory'.format(arg))

        file_names = filter_file_type(arg, type)

        if not len(file_names):
            raise argparse.ArgumentTypeError(
                '{0} contains no {1} files'.format(arg, type)
            )

        return arg, file_names


def is_task_dir_resumeable(
    train_dir, train_dataset, test_dataset, n_train, n_test, sigs, gdml
):
    r"""
    Check if a directory contains `task` and/or `model` files that
    match the configuration of a training process specified in the
    remaining arguments.

    Check if the training and test datasets in each task match
    `train_dataset` and `test_dataset`, if the number of training and
    test points matches and if the choices for the kernel
    hyper-parameter :math:`\sigma` are contained in the list. Check
    also, if the existing tasks/models contain symmetries and if
    that's consistent with the flag `gdml`. This function is useful
    for determining if a training process can be resumed using the
    existing files or not.

    Parameters
    ----------
        train_dir : :obj:`str`
            Path to training directory.
        train_dataset : :obj:`dataset`
            Dataset from which training points are sampled.
        test_dataset : :obj:`test_dataset`
            Dataset from which test points are sampled (may be the
            same as `train_dataset`).
        n_train : int
            Number of training points to sample.
        n_test : int
            Number of test points to sample.
        sigs : :obj:`list` of int
            List of :math:`\sigma` kernel hyper-parameter choices
            (usually: the hyper-parameter search grid)
        gdml : bool
            If `True`, don't include any symmetries in model (GDML),
            otherwise do (sGDML).

    Returns
    -------
        bool
            False, if any of the files in the directory do not match
            the training configuration.
    """

    for file_name in sorted(os.listdir(train_dir)):
        if file_name.endswith('.npz'):
            file_path = os.path.join(train_dir, file_name)
            file = np.load(file_path, allow_pickle=True)

            if 'type' not in file:
                continue
            elif file['type'] == 't' or file['type'] == 'm':

                if (
                    file['md5_train'] != train_dataset['md5']
                    or file['md5_valid'] != test_dataset['md5']
                    or len(file['idxs_train']) != n_train
                    or len(file['idxs_valid']) != n_test
                    or gdml
                    and file['perms'].shape[0] > 1
                    or file['sig'] not in sigs
                ):
                    return False

    return True


### ARGUMENT VALIDATION


def is_strict_pos_int(arg):
    """
    Validate strictly positive integer input.

    Parameters
    ----------
        arg : :obj:`str`
            Integer as string.

    Returns
    -------
        int
            Parsed integer.

    Raises
    ------
        ArgumentTypeError
            If integer is not > 0.
    """
    x = int(arg)
    if x <= 0:
        raise argparse.ArgumentTypeError('must be strictly positive')
    return x


def parse_list_or_range(arg):
    """
    Parses a string that represents either an integer or a range in
    the notation ``<start>:<step>:<stop>``.

    Parameters
    ----------
        arg : :obj:`str`
            Integer or range string.

    Returns
    -------
        int or :obj:`list` of int

    Raises
    ------
        ArgumentTypeError
            If input can neither be interpreted as an integer nor a valid range.
    """

    if re.match(r'^\d+:\d+:\d+$', arg) or re.match(r'^\d+:\d+$', arg):
        rng_params = list(map(int, arg.split(':')))

        step = 1
        if len(rng_params) == 2:  # start, stop
            start, stop = rng_params
        else:  # start, step, stop
            start, step, stop = rng_params

        rng = list(range(start, stop + 1, step))  # include last stop-element in range
        if len(rng) == 0:
            raise argparse.ArgumentTypeError('{0} is an empty range'.format(arg))

        return rng
    elif re.match(r'^\d+$', arg):
        return int(arg)

    raise argparse.ArgumentTypeError(
        '{0} is neither a integer list, nor valid range in the form <start>:[<step>:]<stop>'.format(
            arg
        )
    )

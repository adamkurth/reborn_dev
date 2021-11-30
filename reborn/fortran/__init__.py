# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import hashlib
import importlib
import logging
import numpy.f2py

fortran_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(fortran_path)
std_args = ' -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
std_args += ' -DDISABLE_WARNING_PUSH=-Wunused-function'
std_args += ' -DNPY_DISTUTILS_APPEND_FLAGS=0'
omp_args = " --f90flags='-fopenmp -O2' -lgomp"
os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'
os.environ['NPY_NO_DEPRECATED_API'] = 'NPY_1_7_API_VERSION'


# FIXME: logger and debug should come from utils, but utils imports fortran, which would make circular import...
# FIXME: reborn.utils should not import reborn.fortran.
logger = logging.getLogger()


def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


def check_file_md5(file_path, md5_path=None):
    r"""
    Utility for checking if a file has been modified from a previous version.

    Given a file path, check for a file with ".md5" appended to the path.  If it exists, check if the md5 hash
    saved in the file matches the md5 hash of the current file and return True.  Otherwise, return False.

    Arguments:
        file_path (str): Path to the file.
        md5_path (str): Optional path to the md5 file.

    Returns:
        bool
    """
    if md5_path is None:
        md5_path = file_path+'.md5'
    if not os.path.exists(md5_path):
        return False
    with open(md5_path, 'r') as f:
        md5 = f.readline()
    with open(file_path, 'rb') as f:
        hasher = hashlib.md5()
        hasher.update(f.read())
        new_md5 = str(hasher.hexdigest())
    debug(md5_path, 'md5', md5)
    debug(file_path, 'md5', new_md5)
    if new_md5 == md5:
        return True
    return False


def write_file_md5(file_path, md5_path=None):
    r"""
    Save the md5 hash of a file.  The output will be the same as the original file but with a '.md5' extension appended.

    Arguments:
        file_path (str): The path of the file to make an md5 from.
        md5_path (str): Optional path to the md5 file.

    Returns:
        str: the md5
    """
    md5_path = file_path+'.md5'
    with open(file_path, 'rb') as f:
        hasher = hashlib.md5()
        hasher.update(f.read())
        md5 = str(hasher.hexdigest())
    with open(md5_path, 'w') as f:
        f.write(md5)
    return md5_path


def import_f90(source_file, extra_args='', hash=False, verbose=False, with_omp=False):
    r"""
    Import an piece of fortran code directly to a python module using f2py.  The name of the python module is the same
    as the source file but with the '.f90' extension replaced with '_f'

    Args:
        source_file (str): Name of the fortran module (without the '_f' suffix)
        extra_args (str): Extra arguments for the fortran compiler (e.g. openmp)
        hash (bool): If True, create an md5 hash of the source and recompile only if the source has been modified.
                          Default: False.
        verbose (bool): Print stuff.
        with_omp (bool): Attempt to add the usual OMP flags to the compiler.

    Returns:
        Python module
    """
    extra_args += std_args
    if with_omp:
        extra_args += omp_args
    # First ensure that the source file ends with .f90 and that it is a full path.
    if source_file.split('.')[-1] != 'f90':  # possibly/some/path/name.f90
        source_file += '.f90'
    debug('source_file input', source_file)
    if not os.path.exists(source_file):  # Search for the file in the usual places
        paths = sys.path
        paths.insert(0, os.getcwd())  # Add current directory to path
        for path in paths:
            fn = os.path.join(path, source_file)
            debug(fn)
            if os.path.exists(fn):
                source_file = fn  # /the/full/path/name.90
    if not os.path.exists(source_file):
        raise ValueError("Source file not found", source_file)
    source_file = os.path.abspath(source_file)
    debug('source_file modified', source_file)
    cwd = os.getcwd()
    directory = os.path.dirname(source_file)  # where the source file is located
    sys.path.append(directory)
    os.chdir(directory)  # We move into the directory where the f90 file is located.
    debug('working cwd', os.getcwd())
    source_file = os.path.basename(source_file)
    # We will name the output module just as the input source, but with a _f appended to it.
    modulename = source_file.replace('.f90', '_f')
    debug('module_name', modulename)
    do_compile = True
    # If we are hashing the source file, check if the source has changed
    if hash:
        md5_check = check_file_md5(source_file)
        if md5_check:  # Then the source has not changed and no need to compile
            do_compile = False
        debug('md5_check', md5_check)
    source = open(source_file, "rb").read()
    if do_compile:
        numpy.f2py.compile(source, modulename=modulename, extension='.f90', extra_args=extra_args, verbose=verbose)
    debug('importing...', modulename)
    try:
        module = importlib.import_module(modulename)
    except:
        numpy.f2py.compile(source, modulename=modulename, extension='.f90', extra_args=extra_args, verbose=verbose)
        debug('try again', os.getcwd())
        module = importlib.import_module(modulename)
    debug('module', module)
    if hash:
        write_file_md5(source_file)
    debug('return to ', cwd)
    os.chdir(cwd)
    return module


utils_f = import_f90('utils', hash=True)
interpolations_f = import_f90('interpolations', hash=True)
fortran_indexing_f = import_f90('fortran_indexing', hash=True)
peaks_f = import_f90('peaks', extra_args=omp_args, hash=True)
omp_test_f = import_f90('omp_test', extra_args=omp_args, hash=True)
density_f = import_f90('density', extra_args=omp_args, hash=True)
scatter_f = import_f90('scatter', extra_args=omp_args, hash=True)

from . import omp_test

# try:
#     check_hash(os.path.join(fortran_path, 'utils.f90'))
#     from . import utils_f
# except ImportError:
#     compile_f90('utils.f90')
#     from . import utils_f
#     write_hash(os.path.join(fortran_path, 'utils.f90'))
#
# try:
#     check_hash(os.path.join(fortran_path, 'interpolations.f90'))
#     from . import interpolations_f
# except ImportError:
#     compile_f90('interpolations.f90')
#     from . import interpolations_f
#     write_hash(os.path.join(fortran_path, 'interpolations.f90'))
#
# try:
#     check_hash(os.path.join(fortran_path, 'fortran_indexing.f90'))
#     from . import fortran_indexing_f
# except ImportError:
#     compile_f90('fortran_indexing.f90')
#     from . import fortran_indexing_f
#     write_hash(os.path.join(fortran_path, 'fortran_indexing.f90'))
#
# try:
#     check_hash(os.path.join(fortran_path, 'peaks.f90'))
#     from . import peaks_f
# except ImportError:
#     try:
#         # Attempt to use openmp if it is available
#         compile_f90('peaks.f90', extra_args="--f90flags='-fopenmp -O2' -lgomp")
#         from . import peaks_f
#     except ImportError:
#         compile_f90('peaks.f90')
#         from . import peaks_f
#     write_hash(os.path.join(fortran_path, 'peaks.f90'))
#
# try:
#     check_hash(os.path.join(fortran_path, 'omp_test.f90'))
#     from . import omp_test_f
# except ImportError:
#     try:
#         # Attempt to use openmp if it is available
#         compile_f90('omp_test.f90', extra_args="--f90flags='-fopenmp -O2' -lgomp")
#         from . import omp_test_f
#     except ImportError:
#         compile_f90('omp_test.f90')
#         from . import omp_test_f
#     write_hash(os.path.join(fortran_path, 'omp_test.f90'))
#
# try:
#     check_hash(os.path.join(fortran_path, 'density.f90'))
#     from . import density_f
# except ImportError:
#     compile_f90('density.f90')
#     from . import density_f
#     write_hash(os.path.join(fortran_path, 'density.f90'))

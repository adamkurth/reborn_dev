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
import importlib
import numpy.f2py
from ..utils import debug, check_file_md5, write_file_md5

fortran_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(fortran_path)
std_args = ' -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
std_args += ' -DDISABLE_WARNING_PUSH=-Wunused-function'
std_args += ' -DNPY_DISTUTILS_APPEND_FLAGS=0'
omp_args = " --f90flags='-fopenmp -O2' -lgomp"
os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'
os.environ['NPY_NO_DEPRECATED_API'] = 'NPY_1_7_API_VERSION'


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

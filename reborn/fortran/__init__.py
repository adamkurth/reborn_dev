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
from glob import glob
import numpy.f2py

fortran_path = os.path.split(os.path.abspath(__file__))[0]
# sys.path.append(fortran_path)  # Act of desperation caused by Python's ridiculous import system...

os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'
os.environ['NPY_NO_DEPRECATED_API'] = 'NPY_1_7_API_VERSION'


def check_hash(file_path):
    r"""
    Create an md5 hash of a file.  Note that it searches specifically for files with the .f90 extension.
    Arguments:
        file_path (str): Path to the fortran (.f90) file.
    Returns:
        bool
    """
    if os.path.exists(file_path+'.md5'):
        with open(file_path+'.md5', 'r') as afile:
            old_md5 = afile.readline()
    else:
        old_md5 = None
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher = hashlib.md5()
        hasher.update(buf)
        new_md5 = str(hasher.hexdigest())
    if old_md5 is None:
        match = False
    else:
        match = old_md5 == new_md5
    if match is False:
        compiled = glob(file_path.split('.f90')[0]+'_f*')
        for f in compiled:
            os.remove(f)
    return match


def write_hash(file_path_in):
    r"""
    Write a file that has the md5 hash of another file.  The '.md5' extension is appended to the file name.
    Arguments:
        file_path_in (str): The path of the file to make an md5 from.
    Returns:
        str: the md5
    """
    file_path_save = file_path_in+'.md5'
    with open(file_path_in, 'rb') as afile:
        buf = afile.read()
        hasher = hashlib.md5()
        hasher.update(buf)
        new_md5 = str(hasher.hexdigest())
    with open(file_path_save, 'w') as afile:
        afile.write(new_md5)
    return new_md5


def compile_f90(f90_file, extra_args=''):
    r"""
    Helper function for compiling fortran (.f90) code.
    Arguments:
        f90_file (str): The fortran file to compile.
        extra_args (str): Extra arguments for the fortran compiler (e.g. openmp)
    Returns:
        None
    """
    # numpy.f2py.run_main('-c',os.path.join(pth, f90_file),'-m',f90_file.replace('.f90', '_f'),extra_args)
    numpy.f2py.compile(open(os.path.join(fortran_path, f90_file), "rb").read(),
                       modulename=f90_file.replace('.f90', '_f'),
                       extension='.f90',
                       extra_args=extra_args+' -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -DDISABLE_WARNING_PUSH=-Wunused-function', # -DNPY_DISTUTILS_APPEND_FLAGS=0',
                       verbose=False)
    files = glob('*%s*' % (f90_file.replace('.f90', '_f'),))
    for f in files:
        os.rename(f, os.path.join(fortran_path, os.path.basename(f)))


def import_f90(name, extra_args='', hash_file=False):
    r"""
    Import an piece of fortran code directly to a python module.  This attempts to be a super convenient, one-step
    process that will compile your code if necessary, or just use the pre-compiled code.  If the source code changes,
    the code should re-compile (we create md5 hashes of the source files).

    Of course, with this level of convenience, there are limitations.  So far, this has only been tested in cases
    that involve a single fortran file, and that contain only subroutines.  If you have multiple files that need to
    be compiled and linked, this scheme probably won't work (but we could think about how to make it better...).

    Args:
        name (str): Name of the fortran module (without the '_f' suffix)

    Returns:
        The module.
    """
    source_file_path = os.path.join(fortran_path, name + '.f90')
    if hash_file:
        check_hash(source_file_path)  # This will delete compiled code if the source has changed.
    try:
        module = importlib.import_module('.'+name+'_f', package=__package__)  # Fail if code is not yet compiled.
    except ImportError:
        compile_f90(name+'.f90', extra_args=extra_args)
        module = importlib.import_module('.'+name+'_f', package=__package__)
        if hash_file:
            write_hash(source_file_path)  # Create the md5 hash of the file, so we'll know if it changes in the future.
    return module


omp_args = "--f90flags='-fopenmp -O2' -lgomp"
utils_f = import_f90('utils', hash_file=True)
interpolations_f = import_f90('interpolations', hash_file=True)
fortran_indexing_f = import_f90('fortran_indexing', hash_file=True)
peaks_f = import_f90('peaks', extra_args=omp_args, hash_file=True)
omp_test_f = import_f90('omp_test', extra_args=omp_args, hash_file=True)
density_f = import_f90('density', extra_args=omp_args, hash_file=True)
scatter_f = import_f90('scatter', extra_args=omp_args, hash_file=True)

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

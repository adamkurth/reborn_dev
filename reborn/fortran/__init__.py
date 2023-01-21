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
import glob
import importlib
import numpy.f2py
from ..utils import debug, check_file_md5, write_file_md5
from ..config import configs

fortran_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(fortran_path)
std_args = ' -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
std_args += ' -DDISABLE_WARNING_PUSH=-Wunused-function'
std_args += ' -DNPY_DISTUTILS_APPEND_FLAGS=0'
omp_args = " --f90flags='-fopenmp -O2' -lgomp"
os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'
os.environ['NPY_NO_DEPRECATED_API'] = 'NPY_1_7_API_VERSION'
autocompile = configs['autocompile_fortran']


def import_f90(source_file, extra_args='', hash=True, verbose=False, with_omp=False, autocompile=True):
    r"""
    Import fortran code directly to a python module using f2py.  The name of the python module is the same
    as the source file but with the '.f90' extension replaced with '_f'.  Effectively, we change to the directory
    of the source and compile there.  With the hash option set to True, we first check if the source code has changed
    since the last compile, which avoids recompiling if unnecessary, and also forces a recompile if the source changes
    inadvertently.

    Args:
        source_file (str): Name of the fortran module (without the '_f' suffix)
        extra_args (str): Extra arguments for the fortran compiler (e.g. openmp)
        hash (bool): Create md5 hash of the source, only recompile if the source has changed.  Default: True.
        verbose (bool): Print stuff.
        with_omp (bool): Attempt to add the usual OMP flags to the compiler.
        autocompile (bool): Set to False if you want to compile yourself.  Then

    Returns:
        Python module
    """
    # extra_args get passed to f2py
    extra_args += std_args
    if with_omp:
        extra_args += omp_args
    # First ensure that the source file ends with .f90 and that it is a full path.
    if source_file.split('.')[-1] != 'f90':  # possibly/some/path/name.f90
        source_file += '.f90'
    debug('Input source file:', source_file)
    # If the source file is found immediately, we're done searching.  Else, cast a bigger net.
    if not os.path.exists(source_file):  # Search for the file in the usual places
        paths = sys.path
        debug(os.getcwd(), paths)
        if os.getcwd() not in paths:
            paths.insert(0, os.getcwd())  # Prepend current directory to path.
        for path in paths:
            fn = os.path.join(path, source_file)
            debug('Checking path:', fn)
            if os.path.exists(fn):
                source_file = fn  # /the/full/path/name.90
                debug('Found file!')
                break
    # Now, try again to find the source file
    if not os.path.exists(source_file):
        raise ValueError("Source file not found", source_file)
    source_file = os.path.abspath(source_file)
    debug('source_file =', source_file)
    # We compile in the directory of the source by default so that the binaries are in the same directory.
    # We can make an option to do this differently if need be.
    cwd = os.getcwd()
    directory = os.path.dirname(source_file)  # where the source file is located
    sys.path.append(directory)
    debug('CD to', directory)
    os.chdir(directory)  # We move into the directory where the f90 file is located.
    debug('Working directory', os.getcwd())
    source_file = os.path.basename(source_file)
    # We will name the output module just as the input source, but with a _f appended to it.
    modulename = source_file.replace('.f90', '_f')
    debug('module_name =', modulename)
    # If not auto-compiling, just try to import and fail if import doesn't work
    if not autocompile:
        debug('autocompile off')
        return importlib.import_module(modulename)
    # By default we will not recompile the source unless needed.
    do_compile = False
    # If we are hashing the source file, check if the source has changed
    if hash:
        md5_check = check_file_md5(source_file)
        if not md5_check:  # Then the source has not changed and no need to compile
            do_compile = True
        debug('md5_check', md5_check)
    source = open(source_file, "rb").read()
    compile_args = {'modulename': modulename, 'extension': '.f90', 'extra_args': extra_args,
                    'verbose': verbose}#, 'full_output': False}
    if do_compile:
        bins = glob.glob(modulename+'.cpython*')
        if len(bins) > 0:
            for b in bins:
                print('Removing', b)
                os.remove(b)
        print('Compiling...', source_file, 'to', modulename)
        fp = numpy.f2py.compile(source, **compile_args)
        # print(('='*40+'\n')*5)
        # import inspect
        # for m in inspect.getmembers(fp, predicate=inspect.ismethod):
        #     print(m)
        # print(fp)#str(fp.stdout.decode('utf-8')))#['stdout'])
    # We try to import now, and if it fails, we try to compile one last time.
    try:
        debug('Importing...', modulename)
        module = importlib.import_module(modulename)
    except ImportError:  # This should not be necessary.  Remove it?
        debug('Import failed!')
        debug('Compiling again...', modulename)
        fp = numpy.f2py.compile(source, **compile_args)
        debug('Importing again', modulename)
        module = importlib.import_module(modulename)
    debug('Module name:', module)
    if hash:  # Create a file with the md5 hash of the source f90 file.  This is only created upon successful compile.
        write_file_md5(source_file)
    debug('Return to CWD:', cwd)
    # Finally, back to the original directory.  I *think* this is not necessary.
    os.chdir(cwd)
    return module

dir = os.path.dirname(__file__)
utils_f = import_f90(os.path.join(dir, 'utils'), autocompile=autocompile)
interpolations_f = import_f90(os.path.join(dir, 'interpolations'), autocompile=autocompile)
fortran_indexing_f = import_f90(os.path.join(dir, 'fortran_indexing'), autocompile=autocompile)
peaks_f = import_f90(os.path.join(dir, 'peaks'), autocompile=autocompile, extra_args=omp_args)
omp_test_f = import_f90(os.path.join(dir, 'omp_test'), autocompile=autocompile, extra_args=omp_args)
density_f = import_f90(os.path.join(dir, 'density'), autocompile=autocompile, extra_args=omp_args)
scatter_f = import_f90(os.path.join(dir, 'scatter'), autocompile=autocompile, extra_args=omp_args)
polar_f = import_f90(os.path.join(dir, 'polar'), autocompile=autocompile, extra_args=omp_args)
crystal_f = import_f90(os.path.join(dir, 'crystal'), autocompile=autocompile, extra_args=omp_args)

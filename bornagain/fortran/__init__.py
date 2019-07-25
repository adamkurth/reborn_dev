import numpy.f2py
import os
from glob import glob
pth = os.path.split(os.path.abspath(__file__))[0]


def compile_f90(f90_file, extra_args=''):
    print('Attempting to compile Fortran code.  If this fails, see the docs: https://rkirian.gitlab.io/bornagain')
    numpy.f2py.compile(open(os.path.join(pth, f90_file), "r").read(), modulename=f90_file.replace('.f90', '_f'),
                       extension='.f90', extra_args=extra_args)
    fils = glob('*%s*' % (f90_file.replace('.f90', '_f'),))
    for fil in fils:
        os.rename(fil, os.path.join(pth, os.path.basename(fil)))

try:
    from . import interpolations_f
except ImportError:
    numpy.f2py.compile(open(os.path.join(pth, "interpolations.f90"), "r").read(), modulename='interpolations_f',
                       extension='.f90')
    fils = glob('*interpolations_f*')
    for fil in fils:
        os.rename(fil, os.path.join(pth, os.path.basename(fil)))
    from . import interpolations_f

try:
    from . import wtf_f
except ImportError:
    compile_f90('wtf.f90')
    from . import wtf_f

try:
    from . import peaks_f
except ImportError:
    numpy.f2py.compile(open(os.path.join(pth, "peaks.f90"), "r").read(), modulename='peaks_f', extension='.f90',
                       extra_args="--f90flags='-fopenmp -O2' -lgomp")
    fils = glob('*peaks_f*')
    for fil in fils:
        os.rename(fil, os.path.join(pth, os.path.basename(fil)))
    from . import peaks_f

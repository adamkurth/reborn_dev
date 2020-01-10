import numpy.f2py
import os
from glob import glob
pth = os.path.split(os.path.abspath(__file__))[0]


def compile_f90(f90_file, extra_args=''):
    print('Attempting to compile Fortran code %s.  If this fails, see the docs: https://rkirian.gitlab.io/bornagain'
          % (f90_file,))
    numpy.f2py.compile(open(os.path.join(pth, f90_file), "r").read(), modulename=f90_file.replace('.f90', '_f'),
                       extension='.f90', extra_args=extra_args+' -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
                       verbose=False)
    fils = glob('*%s*' % (f90_file.replace('.f90', '_f'),))
    for fil in fils:
        os.rename(fil, os.path.join(pth, os.path.basename(fil)))


try:
    from . import interpolations_f
except ImportError:
    compile_f90('interpolations.f90')
    from . import interpolations_f

try:
    from . import wtf_f
except ImportError:
    compile_f90('wtf.f90')
    from . import wtf_f

try:
    from . import peaks_f
except ImportError:
    compile_f90('peaks.f90', extra_args="--f90flags='-fopenmp -O2' -lgomp")
    from . import peaks_f

try:
    from . import density_f
except ImportError:
    compile_f90('density.f90')
    from . import density_f

from types import ModuleType
import fortran

utils_f = fortran.import_f90('test/utils.f90', hash=True, verbose=True)
assert(isinstance(utils_f, ModuleType))

utils_f = fortran.import_f90('utils.f90', hash=True, verbose=True)
assert(isinstance(utils_f, ModuleType))

from types import ModuleType
from reborn import fortran


def test_01():
    assert(isinstance(fortran.utils_f, ModuleType))

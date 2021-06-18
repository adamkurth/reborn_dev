import numpy as np
from . import omp_test_f


def get_max_threads():
    n = np.zeros(1, dtype=np.int32)
    omp_test_f.get_max_threads(n)
    return n[0]

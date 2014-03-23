
from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos


DTYPE = np.double
ctypedef np.double_t DTYPE_t


def phaseFactor(q,r):
    
    re, im = c_phase_factor(q,r)
    
    return re + 1j*im

def c_phase_factor(np.ndarray[DTYPE_t, ndim=2] q, np.ndarray[DTYPE_t, ndim=2] r):
    
    if q.shape[1] != 3 or r.shape[1] != 3:
        raise ValueError("Only Nx3 arrays allowed")
    
    cdef int nr = r.shape[0]
    cdef int nq = q.shape[0]
    cdef int qi, ri
    cdef np.ndarray re = np.zeros([nq],dtype=DTYPE)
    cdef np.ndarray im = np.zeros([nq],dtype=DTYPE)
    cdef DTYPE_t tr, ti, ta
    cdef DTYPE_t tqx, tqy, tqz
    
    for qi in range(nq):
        tr = 0
        ti = 0
        tqx = q[qi,0]
        tqy = q[qi,1]
        tqz = q[qi,2]
        for ri in range(nr):
            ta = tqx*r[ri,0] + tqy*r[ri,1] + tqz*r[ri,2]
            tr += cos(ta)
            ti += sin(ta)
        re[qi] = tr
        im[qi] = ti
        
    return (re, im)
            
        
    
    
    

def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print b,
        a, b = b, a + b

def primes(int kmax):
    cdef int n, k, i
    cdef int p[1000]
    result = []
    if kmax > 1000:
        kmax = 1000
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result
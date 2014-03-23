
from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos
cimport cython
from cython.parallel import parallel, prange


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t



def c_phase_factor(np.ndarray[DTYPE_t, ndim=2] q, np.ndarray[DTYPE_t, ndim=2] r):
    
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
   
def phaseFactor(q,r):
    
    if q.ndim != 2 or r.ndim != 2:
        raise ValueError("Arrays must be 2 dimensional")
    if q.shape[1] != 3 or r.shape[1] != 3:
        raise ValueError("Only Nx3 arrays allowed")
    
    re, im = c_phase_factor(q,r)
    
    return re + 1j*im

         


            
def c_molecular_form_factor(np.ndarray[DTYPE_t, ndim=2] q, np.ndarray[DTYPE_t, ndim=2] r,
                            np.ndarray[DTYPE_t, ndim=1] fr,np.ndarray[DTYPE_t, ndim=1] fi):
    
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
            cta = cos(ta)
            sta = sin(ta)
            tr += fr[ri]*cta - fi[ri]*sta
            ti += fi[ri]*cta + fr[ri]*sta 
        re[qi] = tr
        im[qi] = ti
        
    return (re, im)   
    
def molecularFormFactor(q,r,f):    
          
    if q.ndim != 2 or r.ndim != 2:
        raise ValueError("Arrays must be 2 dimensional")
    if q.shape[1] != 3 or r.shape[1] != 3:
        raise ValueError("Only Nx3 arrays allowed")
    
    re, im = c_molecular_form_factor(q,r,f.real, f.imag)
    
    return re + 1j*im
    
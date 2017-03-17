
from __future__ import division
import numpy as np
from datashape.coretypes import real
cimport numpy as np
from libc.math cimport sin, cos, pow
cimport cython
#from cython.parallel import parallel, prange

# DTYPE = np.float32
# ctypedef np.float32_t DTYPE_t

REAL = np.float32
ctypedef np.float32_t REAL_t

COMPLEX = np.complex64
ctypedef np.complex64_t COMPLEX_t

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def phase_factor_qrf_c(np.ndarray[REAL_t, ndim=2] q, np.ndarray[REAL_t, ndim=2] r):
#     
#     cdef int nr = r.shape[0]
#     cdef int nq = q.shape[0]
#     cdef int qi, ri
#     cdef np.ndarray re = np.zeros([nq],dtype=REAL)
#     cdef np.ndarray im = np.zeros([nq],dtype=REAL)
#     cdef REAL_t tr, ti, ta
#     cdef REAL_t tqx, tqy, tqz
#     
#     for qi in range(nq):
#         tr = 0
#         ti = 0
#         tqx = q[qi,0]
#         tqy = q[qi,1]
#         tqz = q[qi,2]
#         for ri in range(nr):
#             ta = tqx*r[ri,0] + tqy*r[ri,1] + tqz*r[ri,2]
#             tr += cos(ta)
#             ti += sin(ta)
#         re[qi] = tr
#         im[qi] = ti
#         
#     return (re, im)
# 
# 
# def phase_factor_qrf(q,r):
#     
#     if q.ndim != 2 or r.ndim != 2:
#         raise ValueError("Arrays must be 2 dimensional")
#     if q.shape[1] != 3 or r.shape[1] != 3:
#         raise ValueError("Only Nx3 arrays allowed")
#     # Check that arrays are c contiguous
#     
#     re, im = phase_factor_qrf_c(q.astype(REAL),r.astype(REAL))
#     
#     return re + 1j*im


# def c_phase_factor(np.ndarray[DTYPE_t, ndim=2] q, np.ndarray[DTYPE_t, ndim=2] r):
#     
#     cdef int nr = r.shape[0]
#     cdef int nq = q.shape[0]
#     cdef int qi, ri
#     cdef np.ndarray re = np.zeros([nq],dtype=DTYPE)
#     cdef np.ndarray im = np.zeros([nq],dtype=DTYPE)
#     cdef DTYPE_t tr, ti, ta
#     cdef DTYPE_t tqx, tqy, tqz
#     
#     for qi in range(nq):
#         tr = 0
#         ti = 0
#         tqx = q[qi,0]
#         tqy = q[qi,1]
#         tqz = q[qi,2]
#         for ri in range(nr):
#             ta = tqx*r[ri,0] + tqy*r[ri,1] + tqz*r[ri,2]
#             tr += cos(ta)
#             ti += sin(ta)
#         re[qi] = tr
#         im[qi] = ti
#         
#     return (re, im)
#    
# def phaseFactor(q,r):
#     
#     if q.ndim != 2 or r.ndim != 2:
#         raise ValueError("Arrays must be 2 dimensional")
#     if q.shape[1] != 3 or r.shape[1] != 3:
#         raise ValueError("Only Nx3 arrays allowed")
#     
#     re, im = c_phase_factor(q,r)
#     
#     return re + 1j*im


            
# def c_molecular_form_factor(np.ndarray[DTYPE_t, ndim=2] q, np.ndarray[DTYPE_t, ndim=2] r,
#                             np.ndarray[DTYPE_t, ndim=1] fr,np.ndarray[DTYPE_t, ndim=1] fi):
#     
#     if q.shape[1] != 3 or r.shape[1] != 3:
#         raise ValueError("Only Nx3 arrays allowed")
#     
#     cdef int nr = r.shape[0]
#     cdef int nq = q.shape[0]
#     cdef int qi, ri
#     cdef np.ndarray re = np.zeros([nq],dtype=DTYPE)
#     cdef np.ndarray im = np.zeros([nq],dtype=DTYPE)
#     cdef DTYPE_t tr, ti, ta
#     cdef DTYPE_t tqx, tqy, tqz
#     
#     for qi in range(nq):
#         tr = 0
#         ti = 0
#         tqx = q[qi,0]
#         tqy = q[qi,1]
#         tqz = q[qi,2]
#         for ri in range(nr):
#             ta = tqx*r[ri,0] + tqy*r[ri,1] + tqz*r[ri,2]
#             cta = cos(ta)
#             sta = sin(ta)
#             tr += fr[ri]*cta - fi[ri]*sta
#             ti += fi[ri]*cta + fr[ri]*sta 
#         re[qi] = tr
#         im[qi] = ti
#         
#     return (re, im)   
    
     
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef cysumpa1(np.ndarray[double] A):
#     cdef double tot = 0
#     cdef int i, n=A.size
#     for i in prange(n, nogil=True):
#         tot = 1
#     return tot
#   
# @cython.boundscheck(False)
# @cython.wraparound(False)  
# cpdef cysumpar2(np.ndarray[double] A):
#     cdef double tot = 0
#     cdef int i, n=A.size
#     for i in range(n):
#         tot = 1
#     return tot   

@cython.boundscheck(False)
@cython.wraparound(False)   
cdef void phase_factor_qrf_c(REAL_t* q, int nq, REAL_t* r, int nr,
                                         REAL_t* fr, REAL_t* fi, 
                                         REAL_t* re, REAL_t* im):
     
    cdef int qi, ri
    cdef REAL_t tr, ti, tqx, tqy, tqz
    cdef REAL_t ta, cta, sta
     
    for qi in range(nq):
        tr = 0
        ti = 0
        tqx = q[qi*3]
        tqy = q[qi*3+1]
        tqz = q[qi*3+2]
        for ri in range(nr):
            ta = tqx*r[ri*3] + tqy*r[ri*3+1] + tqz*r[ri*3+2]
            cta = cos(ta)
            sta = sin(ta)
            tr += fr[ri]*cta - fi[ri]*sta
            ti += fi[ri]*cta + fr[ri]*sta 
        re[qi] = tr
        im[qi] = ti
     
    
def phase_factor_qrf(np.ndarray[REAL_t, ndim=2, mode="c"] q,
                        np.ndarray[REAL_t, ndim=2, mode="c"] r,
                        np.ndarray[COMPLEX_t, ndim=1, mode="c"] f):    
            
    if q.ndim != 2 or r.ndim != 2:
        raise ValueError("Arrays must be 2 dimensional")
    if q.shape[1] != 3 or r.shape[1] != 3:
        raise ValueError("Only Nx3 arrays allowed")
      
    cdef int nq = q.shape[0]
    cdef np.ndarray[REAL_t] re = np.zeros(nq,dtype=REAL)
    cdef np.ndarray[REAL_t] im = np.zeros(nq,dtype=REAL)
    cdef np.ndarray[REAL_t] fr = f.real
    cdef np.ndarray[REAL_t] fi = f.imag
    cdef int nr = r.shape[0]
      
    c_molecular_form_factor(&q[0,0], nq, &r[0,0], nr, 
                            &fr[0], &fi[0],
                            &re[0], &im[0])
      
    return re + 1j*im
    
# @cython.boundscheck(False)
# @cython.wraparound(False)   
# cdef void c_mc_q(double* T, double* B,
#                  double lam, double sw, double bd,
#                  double* F, int nF, 
#                  double* S, int nS):
#     
#     cdef int qi, ri
#     cdef DTYPE_t tr, ti, tqx, tqy, tqz
#     cdef DTYPE_t ta, cta, sta
#     
#     for qi in range(nq):
#         tr = 0
#         ti = 0
#         tqx = q[qi*3]
#         tqy = q[qi*3+1]
#         tqz = q[qi*3+2]
#         for ri in range(nr):
#             ta = tqx*r[ri*3] + tqy*r[ri*3+1] + tqz*r[ri*3+2]
#             cta = cos(ta)
#             sta = sin(ta)
#             tr += fr[ri]*cta - fi[ri]*sta
#             ti += fi[ri]*cta + fr[ri]*sta 
#         re[qi] = tr
#         im[qi] = ti


# cdef void c_panel_molecular_form_factor(double* B, double* T, 
#                                         double* F, double* S,
#                                         int nF, int nS, 
#                                         double* r, int nr,
#                                         double* fr, double* fi, 
#                                         double* re, double* im):
#     
#     cdef int fi, si, ri
#     cdef DTYPE_t tr, ti, tqx, tqy, tqz
#     cdef DTYPE_t ta, cta, sta
#     
#     for fi in range(nF):
#         for si in range(nS):
#             tr = 0
#             ti = 0
#             tqx = q[qi*3]
#             tqy = q[qi*3+1]
#             tqz = q[qi*3+2]
#             for ri in range(nr):
#                 ta = tqx*r[ri*3] + tqy*r[ri*3+1] + tqz*r[ri*3+2]
#                 cta = cos(ta)
#                 sta = sin(ta)
#                 tr += fr[ri]*cta - fi[ri]*sta
#                 ti += fi[ri]*cta + fr[ri]*sta 
#             re[qi] = tr
#             im[qi] = ti
    
    
# def molecularFormFactor(q,r,f):    
#           
#     if q.ndim != 2 or r.ndim != 2:
#         raise ValueError("Arrays must be 2 dimensional")
#     if q.shape[1] != 3 or r.shape[1] != 3:
#         raise ValueError("Only Nx3 arrays allowed")
#     
#     re, im = c_molecular_form_factor(q,r,f.real, f.imag)
#     
#     return re + 1j*im
    
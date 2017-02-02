import numpy as np
import pyopencl as cl

def phaseFactor(q, r, f=None):

    '''
    Calculate scattering factors sum_n f_n exp(-i q.r_n)
    Note the minus sign in this definition above ^^^
    Assumes that q and r are Nx3 vector arrays, and f is real (complex to follow later)
    '''

    if f is None:
        return phaseFactor1(q,r)
    elif ~np.iscomplex(f).any():
        return phaseFactorQRFreal(q,r,f.real)
    else:
    	return phaseFactorQRFcomplex(q,r,f)


def phaseFactorQRFcomplex(q, r, f):

    '''
    This assumes that the scattering factors are complex.
    '''

    r = np.array(r, dtype=np.float32, order='C')
    q = np.array(q, dtype=np.float32, order='C')
    freal = np.array(f.real, dtype=np.float32, order='C')
    fimag = np.array(f.imag, dtype=np.float32, order='C')

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    r_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    fr_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=freal)
    fi_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fimag)
    nq = q.shape[0]
    nr = r.shape[0]
    re_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q.nbytes / 3)
    im_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q.nbytes / 3)

    # run each q vector in parallel
    prg = cl.Program(ctx, """
        __kernel void phase_factor(
        __global const float *q, 
        __global const float *r,
        __global const float *freal,
        __global const float *fimag,
        int nr,
        __global float *re, 
        __global float *im)
        {
            int gid = get_global_id(0);
            re[gid] = 0;
            im[gid] = 0;
            float ph = 0;
            float fr_n = 0;
            float fi_n = 0;
            float sinph = 0;
            float cosph = 0;
            float qx = q[gid*3];
            float qy = q[gid*3+1];
            float qz = q[gid*3+2];
            int i;
    
            for (i=0; i < nr; i++){
            	fr_n = freal[i];
            	fi_n = fimag[i];
                ph = qx*r[i*3] + qy*r[i*3+1] + qz*r[i*3+2];
                sinph = sin(ph);
                cosph = cos(ph);
                re[gid] += fr_n*cosph + fi_n*sinph;
                im[gid] += fr_n*sinph + fi_n*cosph;
            }
        }
        """).build()

    phase_factor = prg.phase_factor
    phase_factor.set_scalar_arg_dtypes([None, None, None, None, int, None, None])

    phase_factor(queue, (nq,), None, q_buf, r_buf, fr_buf, fi_buf, nr, re_buf, im_buf)
    re = np.zeros(nq, dtype=np.float32)
    im = np.zeros(nq, dtype=np.float32)

    cl.enqueue_copy(queue, re, re_buf)
    cl.enqueue_copy(queue, im, im_buf)

    ph = re + 1j * im
    return ph


def phaseFactorPAD(r, f, T, F, S, nF, nS, w):

    '''
    This assumes that the scattering factors are complex.
    '''

    r = np.array(r, dtype=np.float32, order='C')
    q = np.array(q, dtype=np.float32, order='C')
    freal = np.array(f.real, dtype=np.float32, order='C')
    fimag = np.array(f.imag, dtype=np.float32, order='C')
    T = np.array(T, dtype=np.float32, order='C')
    F = np.array(F, dtype=np.float32, order='C')
    S = np.array(S, dtype=np.float32, order='C')
    nF = np.int32(nF)
    nS = np.int32(nS)
    w = np.float32(w)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    r_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    fr_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=freal)
    fi_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fimag)
    nq = nF*nS
    nr = r.shape[0]
    re_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q.nbytes / 3)
    im_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q.nbytes / 3)

    # run each q vector in parallel
    prg = cl.Program(ctx, """
        __kernel void phase_factor(
        __global const float *q, 
        __global const float *r,
        __global const float *freal,
        __global const float *fimag,
        int nr,
        __global float *re, 
        __global float *im)
        {
            int gid = get_global_id(0);
            re[gid] = 0;
            im[gid] = 0;
            float ph = 0;
            float fr_n = 0;
            float fi_n = 0;
            float sinph = 0;
            float cosph = 0;
            float qx = q[gid*3];
            float qy = q[gid*3+1];
            float qz = q[gid*3+2];
            int i;
    
            for (i=0; i < nr; i++){
            	fr_n = freal[i];
            	fi_n = fimag[i];
                ph = qx*r[i*3] + qy*r[i*3+1] + qz*r[i*3+2];
                sinph = sin(ph);
                cosph = cos(ph);
                re[gid] += fr_n*cosph + fi_n*sinph;
                im[gid] += fr_n*sinph + fi_n*cosph;
            }
        }
        """).build()

    phase_factor = prg.phase_factor
    phase_factor.set_scalar_arg_dtypes([None, None, None, None, int, None, None])

    phase_factor(queue, (nq,), None, q_buf, r_buf, fr_buf, fi_buf, nr, re_buf, im_buf)
    re = np.zeros(nq, dtype=np.float32)
    im = np.zeros(nq, dtype=np.float32)

    cl.enqueue_copy(queue, re, re_buf)
    cl.enqueue_copy(queue, im, im_buf)

    ph = re + 1j * im
    return ph


def phaseFactorQRFreal(q, r, f):

    '''
    This assumes that the scattering factors are real.
    '''

    r = np.array(r, dtype=np.float32, order='C')
    q = np.array(q, dtype=np.float32, order='C')
    f = np.array(f, dtype=np.float32, order='C')

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    r_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    f_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    nq = q.shape[0]
    nr = r.shape[0]
    re_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q.nbytes / 3)
    im_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q.nbytes / 3)

    # run each q vector in parallel
    prg = cl.Program(ctx, """
        __kernel void phase_factor(
        __global const float *q, 
        __global const float *r,
        __global const float *f,
        int nr,
        __global float *re, 
        __global float *im)
        {
            int gid = get_global_id(0);
            re[gid] = 0;
            im[gid] = 0;
            float ph = 0;
            float fr_n = 0;
            float qx = q[gid*3];
            float qy = q[gid*3+1];
            float qz = q[gid*3+2];
            int i;
    
            for (i=0; i < nr; i++){
            	fr_n = f[i];
                ph = qx*r[i*3] + qy*r[i*3+1] + qz*r[i*3+2];
                re[gid] += fr_n*cos(ph);
                im[gid] += fr_n*sin(ph);
            }
        }
        """).build()

    phase_factor = prg.phase_factor
    phase_factor.set_scalar_arg_dtypes([None, None, None, int, None, None])

    phase_factor(queue, (nq,), None, q_buf, r_buf, f_buf, nr, re_buf, im_buf)
    re = np.zeros(nq, dtype=np.float32)
    im = np.zeros(nq, dtype=np.float32)

    cl.enqueue_copy(queue, re, re_buf)
    cl.enqueue_copy(queue, im, im_buf)

    ph = re + 1j * im
    return ph


def phaseFactor1(q, r):

    '''
    This assumes the scattering factors are all equal to 1.
    '''	

    r = np.array(r, dtype=np.float32, order='C')
    q = np.array(q, dtype=np.float32, order='C')

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    r_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r)
    q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    nq = q.shape[0]
    nr = r.shape[0]
    re_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q.nbytes / 3)
    im_buf = cl.Buffer(ctx, mf.WRITE_ONLY, q.nbytes / 3)

    # run each q vector in parallel
    prg = cl.Program(ctx, """
        __kernel void phase_factor(
        __global const float *q, 
        __global const float *r, 
        int nr,
        __global float *re, 
        __global float *im)
        {
            int gid = get_global_id(0);
            re[gid] = 0;
            im[gid] = 0;
            float ph = 0;
            float qx = q[gid*3];
            float qy = q[gid*3+1];
            float qz = q[gid*3+2];
            int i;
    
            for (i=0; i < nr; i++){
                ph = qx*r[i*3] + qy*r[i*3+1] + qz*r[i*3+2];
                re[gid] += cos(ph);
                im[gid] += sin(ph);
            }
        }
        """).build()

    phase_factor = prg.phase_factor
    phase_factor.set_scalar_arg_dtypes([None, None, int, None, None])

    phase_factor(queue, (nq,), None, q_buf, r_buf, nr, re_buf, im_buf)
    re = np.zeros(nq, dtype=np.float32)
    im = np.zeros(nq, dtype=np.float32)

    cl.enqueue_copy(queue, re, re_buf)
    cl.enqueue_copy(queue, im, im_buf)

    ph = re + 1j * im
    return ph
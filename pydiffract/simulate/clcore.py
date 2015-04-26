import numpy as np
import pyopencl as cl


def phaseFactor(q, r):

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

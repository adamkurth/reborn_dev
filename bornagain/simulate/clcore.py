import numpy as np
import pyopencl as cl
import time 


def buffer_float32(x,ctx):
    x = np.array(x, dtype=np.float32, order='C')
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

def buffer_complex64(x,ctx):
    x = np.array(x, dtype=np.complex64, order='C')
    return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

def vec4(x):
    # Evdidently pyopencl does not deal with 3-vectors very well, so we use 4-vectors
    # and pad with a zero at the end.
    return np.array([x[0],x[1],x[2],0.0],dtype=np.float32)


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
        return phaseFactorQRF(q,r,f)


def phaseFactorQRF(q, r, f):
    
    '''
    This assumes that the scattering factors are complex.
    '''
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    maxGroupSize = 64 #queue.device.max_work_group_size

    mf = cl.mem_flags

    nq = q.shape[0]
    nAtoms = r.shape[0]
    q_buf = buffer_float32(q,ctx)    
    r_buf = buffer_float32(r.flatten(),ctx)
    f_buf = buffer_complex64(f,ctx)
    a_buf = cl.Buffer(ctx, mf.WRITE_ONLY, nq*4*2)
    
    # run each q vector in parallel
    prg = cl.Program(ctx, """
        #define PI2 6.28318530718f
        #define GROUP_SIZE %d
        __kernel void phase_factor(
        __global const float *q,  
        __global const float *r,
        __global const float2 *f,
        __global float2 *a,
        int nAtoms)
        {
            const int gi = get_global_id(0); /* Global index */
            const int li = get_local_id(0);  /* Local group index */
            
            float ph, sinph, cosph;
            float re = 0;
            float im = 0;
            
            // Each global index corresponds to a particular q-vector
            float4 q4 = (float4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
            
            __local float4 rg[GROUP_SIZE];
            __local float2 fg[GROUP_SIZE];
            
            for (int g=0; g<nAtoms; g+=GROUP_SIZE){
            
                // Here we will move a chunk of atoms to local memory.  Each worker in a 
                // group moves one atom.
                int ai = g+li;
                rg[li] = (float4)(0.0f,0.0f,0.0f,0.0f);
                fg[li] = (float2)(0.0f,0.0f);
                if (ai < nAtoms){
                    rg[li] = (float4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
                    fg[li] = f[ai];
                }
                // Don't proceed until **all** members of the group have finished moving 
                // atom information into local memory.
                barrier(CLK_LOCAL_MEM_FENCE);
                
                // We use a local real and imaginary part to avoid floatint point overflow
                float lre=0;
                float lim=0;
                
                // Now sum up the amplitudes from this subset of atoms
                for (int n=0; n < GROUP_SIZE; n++){
                    ph = -dot(q4,rg[n]); 
                    sinph = native_sin(ph);
                    cosph = native_cos(ph);
                    lre += fg[n].x*cosph - fg[n].y*sinph;
                    lim += fg[n].x*sinph + fg[n].y*cosph;
                }
                re += lre;
                im += lim;
                
                // Don't proceed until this subset of atoms are completed.
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            a[gi].x = re;
            a[gi].y = im;
        }""" % maxGroupSize).build()
    
    phase_factor = prg.phase_factor
    phase_factor.set_scalar_arg_dtypes([None, None, None, None, int])
    
    phase_factor(queue, (nq,), (maxGroupSize,), q_buf, r_buf, f_buf, a_buf,nAtoms)
    a = np.zeros(nq, dtype=np.complex64)
    
    cl.enqueue_copy(queue, a, a_buf)
    
    return a


def phaseFactorPAD(r, f, T, F, S, B, nF, nS, w):
    
    '''
    This should simulate detector panels.  More details to follow...
    '''
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    maxGroupSize = 64 #queue.device.max_work_group_size
    mf = cl.mem_flags
    
    def buf_float(x):
        x = np.array(x, dtype=np.float32, order='C')
        return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    
    def buf_complex(x):
        x = np.array(x, dtype=np.complex64, order='C')
        return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    
    def vec4(x):
        # Evdidently pyopencl does not deal with 3-vectors very well, so we use 4-vectors
        # and pad with a zero at the end.
        return np.array([x[0],x[1],x[2],0.0]).astype(np.float32)
    
    # Setup buffers.  This is very fast.  However, we are assuming that we can just load
    # all atoms into memory, which might not be possible...
    r_buf = buf_float(r)
    f_buf = buf_complex(f)
    nPixels = nF*nS
    nAtoms = r.shape[0]
    T = vec4(T) 
    F = vec4(F) 
    S = vec4(S) 
    B = vec4(B) 
    a_buf = cl.Buffer(ctx, mf.WRITE_ONLY, nPixels*4*2)

    # run each q vector in parallel
    prg = cl.Program(ctx, """
        #define PI2 6.28318530718f
        #define GROUP_SIZE %d
        __kernel void clsim(
        __global const float *r,  /* A float3 array does not seem to work in pyopencl.. */
        __global const float2 *f,
        __global float2 *a, 
        int nPixels,
        int nAtoms,
        int nF,
        int nS,
        float w,
        float4 T,
        float4 F,
        float4 S,
        float4 B)
        {
            const int gi = get_global_id(0); /* Global index */
            const int i = gi %% nF;          /* Pixel coordinate i */
            const int j = gi/nF;             /* Pixel coordinate j */
            const int li = get_local_id(0);  /* Local group index */
            
            float ph, sinph, cosph;
            float re = 0;
            float im = 0;
            
            // Each global index corresponds to a particular q-vector
            float4 V = T + i*F + j*S;
            V /= length(V);
            float4 q = (V-B)*PI2/w;
            
            __local float4 rg[GROUP_SIZE];
            __local float2 fg[GROUP_SIZE];
            
            for (int g=0; g<nAtoms; g+=GROUP_SIZE){
            
                // Here we will move a chunk of atoms to local memory.  Each worker in a 
                // group moves one atom.
                int ai = g+li;
                rg[li] = (float4)(0.0f,0.0f,0.0f,0.0f);
                fg[li] = (float2)(0.0f,0.0f);
                if (ai < nAtoms){
                    rg[li] = (float4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
                    fg[li] = f[ai];
                }
                // Don't proceed until **all** members of the group have finished moving 
                // atom information into local memory.
                barrier(CLK_LOCAL_MEM_FENCE);
                
                // We use a local real and imaginary part to avoid floatint point overflow
                float lre=0;
                float lim=0;
                
                // Now sum up the amplitudes from this subset of atoms
                for (int n=0; n < GROUP_SIZE; n++){
                    ph = -dot(q,rg[n]); 
                    sinph = native_sin(ph);
                    cosph = native_cos(ph);
                    lre += fg[n].x*cosph - fg[n].y*sinph;
                    lim += fg[n].x*sinph + fg[n].y*cosph;
                }
                re += lre;
                im += lim;
                
                // Don't proceed until this subset of atoms are completed.
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            a[gi].x = re;
            a[gi].y = im;
        }""" % maxGroupSize).build()

    clsim = prg.clsim
    clsim.set_scalar_arg_dtypes([None,None,None,int,int,int,int,np.float32,None,None,None,None])

    clsim(queue, (nPixels,), (maxGroupSize,), r_buf,f_buf,a_buf,nPixels,nAtoms,nF,nS,w,T,F,S,B)
    a = np.zeros(nPixels, dtype=np.complex64)

    cl.enqueue_copy(queue, a, a_buf)

    return a


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
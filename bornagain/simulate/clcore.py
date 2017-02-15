import numpy as np
import pyopencl as cl

def pad1d(x,n):
    m = len(x)
    return np.concatenate([x,np.zeros([n-m])])

def padVec(x,n):
    assert x.shape[1] == 3
    m = x.shape[0]
    return np.concatenate([x,np.zeros([n-m,3])])

def buffer_read_float32(x,context):
    x = np.array(x, dtype=np.float32, order='C')
    return cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

def buffer_read_complex64(x,context):
    x = np.array(x, dtype=np.complex64, order='C')
    return cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

def vec4(x,dtype=np.float32):
    # Evdidently pyopencl does not deal with 3-vectors very well, so we use 4-vectors
    # and pad with a zero at the end.
    return np.array([x[0],x[1],x[2],0.0],dtype=dtype)


def phaseFactorQRF(q, r, f, context=None):

    '''
    Calculate the diffraction amplitude sum over f*exp(-iq.r) 
    
    Input Arguments:
    q:       Numpy array [N,3] of scattering vectors (2.pi/lambda)
    r:       Numpy array [M,3] of atomic coordinates
    f:       Numpy array [M] of complex scattering factors
    context: Optional pyopencl context cl.create_some_context()
    
    Return:
    A:       Numpy array [N] of complex amplitudes
    '''

    nPixels = q.shape[0]
    nAtoms = r.shape[0]

    if context is None: context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    groupSize = 64 #queue.device.max_work_group_size
    globalSize = np.int(np.ceil(nPixels/np.float(groupSize))*groupSize)
    mf = cl.mem_flags

    q_buf = buffer_read_float32(q,context)
    r_buf = buffer_read_float32(r.flatten(),context)
    f_buf = buffer_read_complex64(f,context)
    a_buf = cl.Buffer(context, mf.WRITE_ONLY, nPixels*4*2)

    # run each q vector in parallel
    prg = cl.Program(context, """
        #define GROUP_SIZE """ + ("%d" % groupSize) + """
        __kernel void phaseFactorQRF_cl(
        __global const float *q,
        __global const float *r,
        __global const float2 *f,
        __global float2 *a,
        int nAtoms,
        int nPixels)
        {
            const int gi = get_global_id(0); /* Global index */
            const int li = get_local_id(0);  /* Local group index */

            float ph, sinph, cosph;
            float re = 0;
            float im = 0;

            // Each global index corresponds to a particular q-vector.  Note that the
            // global index could be larger than the number of pixels because it must be a
            // multiple of the group size.
            float4 q4;
            if (gi < nPixels){
                q4 = (float4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
            } else {
                q4 = (float4)(0.0f,0.0f,0.0f,0.0f);
            }
            __local float4 rg[GROUP_SIZE];
            __local float2 fg[GROUP_SIZE];

            for (int g=0; g<nAtoms; g+=GROUP_SIZE){

                // Here we will move a chunk of atoms to local memory.  Each worker in a
                // group moves one atom.
                int ai = g+li;

                if (ai < nAtoms ){
                    rg[li] = (float4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
                    fg[li] = f[ai];
                } else {
                    rg[li] = (float4)(0.0f,0.0f,0.0f,0.0f);
                    fg[li] = (float2)(0.0f,0.0f);
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
            if (gi < nPixels){
                a[gi].x = re;
                a[gi].y = im;
            }
        }""").build()

    phaseFactorQRF_cl = prg.phaseFactorQRF_cl
    phaseFactorQRF_cl.set_scalar_arg_dtypes(     [ None,  None,  None,  None,  int,    int ]  )
    phaseFactorQRF_cl(queue, (globalSize,), (groupSize,),q_buf, r_buf, f_buf, a_buf, nAtoms, nPixels)
    a = np.zeros(nPixels, dtype=np.complex64)

    cl.enqueue_copy(queue, a, a_buf)

    return a


def phaseFactorPAD(r, f, T, F, S, B, nF, nS, w, context=None):

    '''
    This should simulate detector panels.  More details to follow...
    '''

    nPixels = nF*nS
    nAtoms = r.shape[0]

    if context is None: context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    groupSize = 64 #queue.device.max_work_group_size
    globalSize = np.int(np.ceil(nPixels/np.float(groupSize))*groupSize)
    mf = cl.mem_flags

    def buf_float(x):
        x = np.array(x, dtype=np.float32, order='C')
        return cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)

    def buf_complex(x):
        x = np.array(x, dtype=np.complex64, order='C')
        return cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)

    def vec4(x):
        # Evdidently pyopencl does not deal with 3-vectors very well, so we use 4-vectors
        # and pad with a zero at the end.
        return np.array([x[0],x[1],x[2],0.0]).astype(np.float32)

    # Setup buffers.  This is very fast.  However, we are assuming that we can just load
    # all atoms into memory, which might not be possible...
    r_buf = buf_float(r)
    f_buf = buf_complex(f)
    T = vec4(T)
    F = vec4(F)
    S = vec4(S)
    B = vec4(B)
    a_buf = cl.Buffer(context, mf.WRITE_ONLY, nPixels*4*2)

    # run each q vector in parallel
    prg = cl.Program(context, """
        #define PI2 6.28318530718f
        #define GROUP_SIZE %d
        __kernel void phaseFactorPAD_cl(
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
            float4 V;
            float4 q;

            V = T + i*F + j*S;
            V /= length(V);
            q = (V-B)*PI2/w;

            __local float4 rg[GROUP_SIZE];
            __local float2 fg[GROUP_SIZE];

            for (int g=0; g<nAtoms; g+=GROUP_SIZE){

                // Here we will move a chunk of atoms to local memory.  Each worker in a
                // group moves one atom.
                int ai = g+li;

                if (ai < nAtoms){
                    rg[li] = (float4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
                    fg[li] = f[ai];
                } else {
                    rg[li] = (float4)(0.0f,0.0f,0.0f,0.0f);
                    fg[li] = (float2)(0.0f,0.0f);
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

            if (gi < nPixels ){
                a[gi].x = re;
                a[gi].y = im;
            }

        }""" % groupSize).build()

    phaseFactorPAD_cl = prg.phaseFactorPAD_cl
    phaseFactorPAD_cl.set_scalar_arg_dtypes([None,None,None,int,int,int,int,np.float32,None,None,None,None])

    phaseFactorPAD_cl(queue, (globalSize,), (groupSize,), r_buf,f_buf,a_buf,nPixels,nAtoms,nF,nS,w,T,F,S,B)
    a = np.zeros(nPixels, dtype=np.complex64)

    cl.enqueue_copy(queue, a, a_buf)

    return a

def phaseFactor3DM(r, f, N,Qmin=None, Qmax=None, context=None):

    '''
    This should simulate a regular 3D mesh of q-space samples.
    '''

    assert Qmax is not None

    N = np.array(N,dtype=np.int32)
    Qmax = np.array(Qmax,dtype=np.float32)
    Qmin = np.array(Qmin,dtype=np.float32)

    if len(N.shape) == 0: N = np.ones(3,dtype=np.int32)*N
    if len(Qmax.shape) == 0: Qmax = np.ones(3,dtype=np.float32)*Qmax
    if len(Qmin.shape) == 0: Qmin = np.ones(3,dtype=np.float32)*Qmin

    deltaQ = np.array((Qmax-Qmin)/(N-1.0),dtype=np.float32)

    nAtoms = r.shape[0]
    nPixels = N[0]*N[1]*N[2]

    if context is None: context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    groupSize = 64 #queue.device.max_work_group_size
    globalSize = np.int(np.ceil(nPixels/np.float(groupSize))*groupSize)
    mf = cl.mem_flags

    # Setup buffers.  This is very fast.  However, we are assuming that we can just load
    # all atoms into memory, which might not be possible...
    r_buf = buffer_read_float32(r,context)
    f_buf = buffer_read_complex64(f,context)
    N = vec4(N,dtype=np.int32)
    deltaQ = vec4(deltaQ,dtype=np.float32)
    Qmin = vec4(Qmin,dtype=np.float32)
    a_buf = cl.Buffer(context, mf.WRITE_ONLY, nPixels*4*2)

    # run each q vector in parallel
    prg = cl.Program(context, """
        #define GROUP_SIZE """ + ('%d' % groupSize) + """
        __kernel void phaseFactor3DM_cl(
        __global const float *r,  /* A float3 array does not seem to work in pyopencl.. */
        __global const float2 *f,
        __global float2 *a,
        int nPixels,
        int nAtoms,
        int4 N,
        float4 deltaQ,
        float4 Qmin)
        {
            const int Nxy = N.x*N.y;
            const int gi = get_global_id(0); /* Global index */
            const int i = gi % N.x;          /* Voxel coordinate i (x) */
            const int j = (gi/N.x) % N.y;    /* Voxel coordinate j (y) */
            const int k = gi/Nxy;            /* Voxel corrdinate k (z) */
            const int li = get_local_id(0);  /* Local group index */

            float ph, sinph, cosph;
            float re = 0;
            float im = 0;
            int ai;

            // Each global index corresponds to a particular q-vector
            const float4 q4 = (float4)(i*deltaQ.x+Qmin.x,
                                       j*deltaQ.y+Qmin.y,
                                       k*deltaQ.z+Qmin.z,0.0f);

            __local float4 rg[GROUP_SIZE];
            __local float2 fg[GROUP_SIZE];

            for (int g=0; g<nAtoms; g+=GROUP_SIZE){

                // Here we will move a chunk of atoms to local memory.  Each worker in a
                // group moves one atom.
                ai = g+li;
                if (ai < nAtoms){
                    rg[li] = (float4)(r[ai*3],r[ai*3+1],r[ai*3+2],0.0f);
                    fg[li] = f[ai];
                } else {
                    rg[li] = (float4)(0.0f,0.0f,0.0f,0.0f);
                    fg[li] = (float2)(0.0f,0.0f);
                }
                // Don't proceed until **all** members of the group have finished moving
                // atom information into local memory.
                barrier(CLK_LOCAL_MEM_FENCE);

                // We use a local real and imaginary part to avoid floating point overflow
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

            if (gi < nPixels){
                a[gi].x = re;
                a[gi].y = im;
            }
        }""").build()

    #print(globalSize,groupSize)

    phaseFactor3DM_cl = prg.phaseFactor3DM_cl
    phaseFactor3DM_cl.set_scalar_arg_dtypes([None,None,None,int,int,None,None,None])

    phaseFactor3DM_cl(queue, (globalSize,), (groupSize,), r_buf,f_buf,a_buf,nPixels,nAtoms,N,deltaQ,Qmin)
    a = np.zeros(nPixels, dtype=np.complex64)

    cl.enqueue_copy(queue, a, a_buf)

    return a


class PatternGenerator(object):

    def __init__(self, A, N, Qmin=None, Qmax=None, context=None):
        
        
        
         

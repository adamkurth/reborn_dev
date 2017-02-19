import numpy as np
import pyopencl as cl

def clbuffer_readonly_float32(x,context):
    x = np.array(x, dtype=np.float32, order='C')
    return cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

def clbuffer_readonly_complex64(x,context):
    x = np.array(x, dtype=np.complex64, order='C')
    return cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

def vec4(x,dtype=np.float32):
    # Evdidently pyopencl does not deal with 3-vectors very well, so we use 4-vectors
    # and pad with a zero at the end.
    return np.array([x[0],x[1],x[2],0.0],dtype=dtype)


def phase_factor_qrf(q, r, f, context=None, group_size=64):

    '''
    Calculate the diffraction amplitude sum over n: f_n*exp(-iq.r_n)
    
    Input:
    q:       Numpy array [N,3] of scattering vectors (2.pi/lambda)
    r:       Numpy array [M,3] of atomic coordinates
    f:       Numpy array [M] of complex scattering factors
    context: Optional pyopencl context cl.create_some_context()
    group_size: Optional specification of pyopencl group size
    
    Return:
    A:       Numpy array [N] of complex amplitudes
    '''

    nPixels = q.shape[0]
    nAtoms = r.shape[0]

    if context is None: context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    groupSize = group_size
    if groupSize > queue.device.max_work_group_size:
        groupSize = queue.device.max_work_group_size
    globalSize = np.int(np.ceil(nPixels/np.float(groupSize))*groupSize)
    mf = cl.mem_flags

    q_buf = clbuffer_readonly_float32(q,context)
    r_buf = clbuffer_readonly_float32(r.flatten(),context)
    f_buf = clbuffer_readonly_complex64(f,context)
    a_buf = cl.Buffer(context, mf.WRITE_ONLY, nPixels*4*2)

    # run each q vector in parallel
    prg = cl.Program(context, """
        #define GROUP_SIZE """ + ("%d" % groupSize) + """
        __kernel void phase_factor_qrf_cl(
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

    phase_factor_qrf_cl = prg.phase_factor_qrf_cl
    phase_factor_qrf_cl.set_scalar_arg_dtypes(     [ None,  None,  None,  None,  int,    int ]  )
    phase_factor_qrf_cl(queue, (globalSize,), (groupSize,),q_buf, r_buf, f_buf, a_buf, nAtoms, nPixels)
    
    a = np.zeros(nPixels, dtype=np.complex64)
    cl.enqueue_copy(queue, a, a_buf)

    return a


def phase_factor_pad(r, f, T, F, S, B, nF, nS, w, context=None, group_size=64):

    '''
    This should simulate detector panels.  
    
    Input:
    r:       An Nx3 numpy array with atomic coordinates (meters)
    f:       A numpy array with complex scattering factors
    T:       A 1x3 numpy array with vector components pointing from sample to the center 
             of the first pixel in memory
    F:       A 1x3 numpy array containing the basis vector components pointing in the
             direction corresponding to contiguous pixels in memory ("fast scan").
    S:       A 1x3 numpy array containing the basis vector components pointing in the
             direction corresponding to non-contiguous pixels in memory ("slow scan").
    B:       A 1x3 numpy array with unit-vector components corresponding to the incident
             x-ray beam direction
    nF:      Number of fast-scan pixels (corresponding to F vector) in the detector panel
    nS:      Number of slow-scan pixels (corresponding to S vector) in the detector panel
    w:       The photon wavelength in meters
    context: The opencl context may be specified, optionally
    group_size: Optional specification of pyopencl group size
    
    Output:
    A:        A numpy array of length nF*nS containing complex scattering amplitudes 
    '''

    nPixels = nF*nS
    nAtoms = r.shape[0]

    if context is None: context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    groupSize = group_size
    if groupSize > queue.device.max_work_group_size:
        groupSize = queue.device.max_work_group_size
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
        __kernel void phase_factor_pad_cl(
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

    phase_factor_pad_cl = prg.phase_factor_pad_cl
    phase_factor_pad_cl.set_scalar_arg_dtypes([None,None,None,int,int,int,int,np.float32,None,None,None,None])

    phase_factor_pad_cl(queue, (globalSize,), (groupSize,), r_buf,f_buf,a_buf,nPixels,nAtoms,nF,nS,w,T,F,S,B)
    a = np.zeros(nPixels, dtype=np.complex64)

    cl.enqueue_copy(queue, a, a_buf)

    return a

def phase_factor_mesh(r, f, N, q_min, q_max, context=None, copy_buffer=True, group_size=64):

    '''
    This should simulate a regular 3D mesh of q-space samples.
    
    Input:
    r:            An Nx3 numpy array of atomic coordinates (meters)
    f:            A numpy array of complex atomic scattering factors
    N:            A scalar or length-3 array specifying the number of q-space samples in 
                  each of the three dimensions
    q_min:        A scalar or length-3 array specifying the minimum q-space magnitudes in
                  the 3d mesh.  These values specify the center of the voxel.
    q_max:        A scalar or length-3 array specifying the maximum q-space magnitudes in
                  the 3d mesh.  These values specify the center of the voxel.
    context:      Optionally specify the opencl context
    copy_buffer:  Set to False if you wish to keep the output array in GPU memory
    group_size:   Optional specification of pyopencl group size
    
    Output:
    An array of complex scattering amplitudes.  By default this is a normal numpy array.  
    Optionally, this may be an opencl buffer.  
    '''

    N = np.array(N,dtype=np.int32)
    q_max = np.array(q_max,dtype=np.float32)
    q_min = np.array(q_min,dtype=np.float32)

    if len(N.shape) == 0: N = np.ones(3,dtype=np.int32)*N
    if len(q_max.shape) == 0: q_max = np.ones(3,dtype=np.float32)*q_max
    if len(q_min.shape) == 0: q_min = np.ones(3,dtype=np.float32)*q_min

    deltaQ = np.array((q_max-q_min)/(N-1.0),dtype=np.float32)

    nAtoms = r.shape[0]
    nPixels = N[0]*N[1]*N[2]

    if context is None: context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    groupSize = group_size
    if groupSize > queue.device.max_work_group_size:
        groupSize = queue.device.max_work_group_size
    globalSize = np.int(np.ceil(nPixels/np.float(groupSize))*groupSize)
    mf = cl.mem_flags

    # Setup buffers.  This is very fast.  However, we are assuming that we can just load
    # all atoms into memory, which might not be possible...
    r_buf = clbuffer_readonly_float32(r,context)
    f_buf = clbuffer_readonly_complex64(f,context)
    N = vec4(N,dtype=np.int32)
    deltaQ = vec4(deltaQ,dtype=np.float32)
    q_min = vec4(q_min,dtype=np.float32)
    a_buf = cl.Buffer(context, mf.WRITE_ONLY, nPixels*4*2)

    # run each q vector in parallel
    prg = cl.Program(context, """
        #define GROUP_SIZE """ + ('%d' % groupSize) + """
        __kernel void phase_factor_mesh_cl(
        __global const float *r,  /* A float3 array does not seem to work in pyopencl.. */
        __global const float2 *f,
        __global float2 *a,
        int nPixels,
        int nAtoms,
        int4 N,
        float4 deltaQ,
        float4 q_min)
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
            const float4 q4 = (float4)(i*deltaQ.x+q_min.x,
                                       j*deltaQ.y+q_min.y,
                                       k*deltaQ.z+q_min.z,0.0f);

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

    phase_factor_mesh_cl = prg.phase_factor_mesh_cl
    phase_factor_mesh_cl.set_scalar_arg_dtypes([None,None,None,int,int,None,None,None])

    phase_factor_mesh_cl(queue, (globalSize,), (groupSize,), r_buf,f_buf,a_buf,nPixels,nAtoms,N,deltaQ,q_min)
    
    if copy_buffer == True:
        a = np.zeros(nPixels, dtype=np.complex64)
        cl.enqueue_copy(queue, a, a_buf)
        return a
    else:
        return a_buf


def buffer_mesh_lookup(a, N, q_min, q_max, q, context=None, copy_buffer=True, group_size=64):

    """
    This is supposed to lookup intensities from a 3d mesh of amplitudes.
    
    Input:
    a:       Numpy array of complex scattering amplitudes generated from the function
               phase_factor_mesh()
    N:       As defined in phase_factor_mesh()
    q_min:   As defined in phase_factor_mesh()
    q_max:   As defined in phase_factor_mesh()
    q:       An Nx3 numpy array of q-space coordinates at which we want to interpolate 
               the complex amplitudes in a_buf
    context:
    copy_buffer:
    group_size:
               
    Output:
    A numpy array of complex amplitudes.
    """
    
    N = np.array(N,dtype=np.int32)
    q_max = np.array(q_max,dtype=np.float32)
    q_min = np.array(q_min,dtype=np.float32)

    if len(N.shape) == 0: N = np.ones(3,dtype=np.int32)*N
    if len(q_max.shape) == 0: q_max = np.ones(3,dtype=np.float32)*q_max
    if len(q_min.shape) == 0: q_min = np.ones(3,dtype=np.float32)*q_min

    deltaQ = np.array((q_max-q_min)/(N-1.0),dtype=np.float32)

    nPixels = q.shape[0]

    if context is None: context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    groupSize = group_size
    if groupSize > queue.device.max_work_group_size:
        groupSize = queue.device.max_work_group_size
    globalSize = np.int(np.ceil(nPixels/np.float(groupSize))*groupSize)
    mf = cl.mem_flags

    # Setup buffers.  This is very fast.  However, we are assuming that we can just load
    # all atoms into memory, which might not be possible...
    a_buf = clbuffer_readonly_complex64(a,context)
    q_buf = clbuffer_readonly_float32(q,context)
    N = vec4(N,dtype=np.int32)
    deltaQ = vec4(deltaQ,dtype=np.float32)
    q_min = vec4(q_min,dtype=np.float32)
    out_buf = cl.Buffer(context, mf.WRITE_ONLY, nPixels*4*2)
    
    # run each q vector in parallel
    prg = cl.Program(context, """
        #define GROUP_SIZE """ + ('%d' % groupSize) + """
        __kernel void buffer_mesh_lookup_cl(
        __global float2 *a,
        __global float *q,
        __global float2 *out_buf,
        int nPixels,
        int4 N,
        float4 deltaQ,
        float4 q_min)
        {
            const int gi = get_global_id(0); /* Global index is q-vector index */
            const float4 q4 = (float4)(q[gi*3],q[gi*3+1],q[gi*3+2],0.0f);
            
            const float i_f = (q4.x - q_min.x)/deltaQ.x; /* Voxel coordinate i (x) */
            const float j_f = (q4.y - q_min.y)/deltaQ.y; /* Voxel coordinate j (y) */
            const float k_f = (q4.z - q_min.z)/deltaQ.z; /* Voxel corrdinate k (z) */

            const int i = (int)(round(i_f));
            const int j = (int)(round(j_f));
            const int k = (int)(round(k_f));
            
            if (i >= 0 && i < N.x && j >= 0 && j < N.y && k >= 0 && k < N.z){
                const int idx = k*N.x*N.y + j*N.x + i;
                out_buf[gi].x = a[idx].x;
                out_buf[gi].y = a[idx].y;
            } else {
                out_buf[gi].x = 0.0f;
                out_buf[gi].y = 0.0f;
            }

        }""").build()

    buffer_mesh_lookup_cl = prg.buffer_mesh_lookup_cl
    buffer_mesh_lookup_cl.set_scalar_arg_dtypes([None,None,None,int,None,None,None])

    buffer_mesh_lookup_cl(queue, (globalSize,), (groupSize,), a_buf,q_buf,out_buf,nPixels,N,deltaQ,q_min)
    
    if copy_buffer == True:
        out = np.zeros(nPixels, dtype=np.complex64)
        cl.enqueue_copy(queue, out, out_buf)
        return out
    else:
        return None
    


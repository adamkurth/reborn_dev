"""
This module contains some core functions that are useful for simulating
diffraction on GPU devices.  It is not finished yet...

Some environment variables that affect this module:
'BORNAGAIN_CL_GROUPSIZE' : This sets the default groupsize.  It will
otherwise be 32.  If you are using a CPU you might want to set 
BORNAGAIN_CL_GROUPSIZE=1 .
"""

import sys
import os
import pkg_resources

import numpy as np
import pyopencl as cl
import pyopencl.array
#from __builtin__ import None    # this is causing problems for some reason...

clcore_file = pkg_resources.resource_filename(
    'bornagain.simulate', 'clcore.cpp')

default_context = cl.create_some_context()
default_queue = cl.CommandQueue(default_context)

default_group_size = None
if os.environ.get('BORNAGAIN_CL_GROUPSIZE') is not None:
    default_group_size = np.int(os.environ.get('BORNAGAIN_CL_GROUPSIZE'))
if default_group_size is None:
    default_group_size = 32
max_group_size = default_queue.device.max_work_group_size
if default_group_size > max_group_size:
    sys.stderr.write('Changing group size from %d to %d.\n'
                     'Set BORNAGAIN_CL_GROUPSIZE=%d to avoid this error.\n' 
                     % (default_group_size, max_group_size, max_group_size))
    default_group_size = max_group_size

programs = cl.Program(default_context, open(clcore_file).read()).build(
                        options=['-D', 'GROUP_SIZE=%d' % default_group_size])

def vec4(x, dtype=None):
    """
Evdidently pyopencl does not deal with 3-vectors very well, so we use
4-vectors and pad with a zero at the end.

Arguments:
    x: array
    dtype: Optional np.dtype (default np.float32)
    
Returns:
    numpy array of length 4
    """
    
    if dtype is None:
        dtype = np.float32
    return np.array([x[0], x[1], x[2], 0.0], dtype=dtype)


def to_device(array=None, shape=None, dtype=None, queue=None):
    """
This is a thin wrapper for pyopencl.array.to_device().  It will convert a numpy 
array into a pyopencl.array and send it to the device memory.  So far this only
deals with float and comlex arrays, and it should figure out which type it is.

Arguments:
    array (numpy/cl array; float/complex type): Input array.
    shape (tuple): Optionally specify the shape of the desired array.  This is 
                    ignored if array is not None. 
    dtype (np.dtype): Specify the desired type in opencl.  The two types that 
                       are useful here are np.float32 and np.complex64

Returns:
    pyopencl array
    """

    if queue is None:
        queue = default_queue

    if isinstance(array, cl.array.Array):
        return array

    if array is None:
        array = np.zeros(shape, dtype=dtype)

    if dtype is None:
        if np.iscomplexobj(array):
            dtype = np.complex64
        else:
            dtype = np.float32

    return cl.array.to_device(queue, np.ascontiguousarray(array.astype(dtype)))


get_group_size_cl = programs.get_group_size
get_group_size_cl.set_scalar_arg_dtypes([None])

def get_group_size(context=None,queue=None,group_size=None):
    
    if context is None:
        context = default_context
    if queue is None:
        queue = default_queue
    if group_size is None:
        group_size = default_group_size
    
    group_size_dev = to_device(np.zeros((1)), dtype=np.int32, queue=queue)
    
    get_group_size_cl(queue, (group_size,), (group_size,), group_size_dev.data)

    return group_size_dev.get()[0]
    

phase_factor_qrf_cl = programs.phase_factor_qrf
phase_factor_qrf_cl.set_scalar_arg_dtypes(
    [None, None, None, None, None, np.int32, np.int32])


def phase_factor_qrf(q, r, f, R=None, a=None, context=None,
                     queue=None, group_size=None):
    '''
Calculate diffraction amplitudes: sum over f_n*exp(-iq.r_n)

Arguments:
    q (numpy/cl float array [N,3]): Scattering vectors (2\pi/\lambda).
    r (numpy/cl float array [M,3]): Atomic coordinates.
    f (numpy/cl complex array [M]): Complex scattering factors.
    R (numpy array [3,3]): Rotation matrix acting on q vectors.
    a (cl complex array [N]): Optional container for complex scattering 
      amplitudes.
    context (pyopencl context): Optional pyopencl context (e.g. use 
      cl.create_some_context() to create one).
    queue (pyopencl context): Optional pyopencl queue (e.g use 
      cl.CommandQueue(context) to create one).
    group_size (int): Optional pyopencl group size.

Returns:
    (numpy/cl complex array [N]): Diffraction amplitudes.  Will be a cl array 
      if there are input cl arrays.
    '''

    if context is None:
        context = default_context
    if queue is None:
        queue = default_queue
    if group_size is None:
        group_size = default_group_size

    if R is None:
        R = np.eye(3, dtype=np.float32)
    R16 = np.zeros([16], dtype=np.float32)
    R16[0:9] = R.flatten().astype(np.float32)

    n_pixels = np.int32(q.shape[0])
    n_atoms = np.int32(r.shape[0])
    q_dev = to_device(q, dtype=np.float32, queue=queue)
    r_dev = to_device(r, dtype=np.float32, queue=queue)
    f_dev = to_device(f, dtype=np.complex64, queue=queue)
    a_dev = to_device(a, dtype=np.complex64, shape=(n_pixels), queue=queue)

    global_size = np.int(np.ceil(n_pixels / np.float(group_size)) * group_size)

    phase_factor_qrf_cl(queue, (global_size,), (group_size,), q_dev.data,
                        r_dev.data, f_dev.data, R16, a_dev.data, n_atoms,
                        n_pixels)

    if a is None:
        return a_dev.get()
    else:
        return a_dev


phase_factor_pad_cl = programs.phase_factor_pad
phase_factor_pad_cl.set_scalar_arg_dtypes(
    [None, None, None, None, np.int32, np.int32, np.int32, np.int32,
     np.float32, None, None, None, None])


def phase_factor_pad(r, f, T, F, S, B, nF, nS, w, R=None,
                     a=None, context=None,
                     queue=None, group_size=None):
    '''
This should simulate detector panels.

Arguments:
    r: An Nx3 numpy array with atomic coordinates (meters)
    f: A numpy array with complex scattering factors
    T: A 1x3 numpy array with vector components pointing from sample to
       the center of the first pixel in memory
    F: A 1x3 numpy array containing the basis vector components pointing
        in the direction corresponding to contiguous pixels in memory
        ("fast scan").
    S: A 1x3 numpy array containing the basis vector components pointing
        in the direction corresponding to non-contiguous pixels in
        memory ("slow scan").
    B: A 1x3 numpy array with unit-vector components corresponding to the
        incident x-ray beam direction
    nF: Number of fast-scan pixels (corresponding to F vector) in the
        detector panel
    nS: Number of slow-scan pixels (corresponding to S vector) in the
        detector panel
    w: The photon wavelength in meters
    R: Optional numpy array [3x3] specifying rotation of q vectors
    a: Optional output complex scattering amplitude cl array
    context: Optional pyopencl context [cl.create_some_context()]
    queue:   Optional pyopencl queue [cl.CommandQueue(context)]
    group_size: Optional specification of pyopencl group size (default 64 or
       maximum)

Returns:
    A: A numpy array of length nF*nS containing complex scattering
amplitudes
    '''

    if context is None:
        context = default_context
    if queue is None:
        queue = default_queue
    if group_size is None:
        group_size = default_group_size

    if R is None:
        R = np.eye(3, dtype=np.float32)
    R16 = np.zeros([16], dtype=np.float32)
    R16[0:9] = R.flatten().astype(np.float32)

    nF = np.int32(nF)
    nS = np.int32(nS)
    n_pixels = np.int32(nF * nS)
    n_atoms = np.int32(r.shape[0])
    r_dev = to_device(r, dtype=np.float32, queue=queue)
    f_dev = to_device(f, dtype=np.complex64, queue=queue)
    T = vec4(T)
    F = vec4(F)
    S = vec4(S)
    B = vec4(B)
    a_dev = to_device(a, dtype=np.complex64, shape=(n_pixels), queue=queue)

    global_size = np.int(np.ceil(n_pixels / np.float(group_size)) * group_size)

    phase_factor_pad_cl(queue, (global_size,), (group_size,), r_dev.data,
                        f_dev.data, R16, a_dev.data, n_pixels, n_atoms, nF, nS,
                        w, T, F, S, B)

    if a is None:
        return a_dev.get()
    else:
        return a_dev

phase_factor_mesh_cl = programs.phase_factor_mesh
phase_factor_mesh_cl.set_scalar_arg_dtypes(
    [None, None, None, np.int32, np.int32, None, None, None])


def phase_factor_mesh(r, f, N, q_min, q_max, a=None, context=None,
                     queue=None, group_size=None):
    '''
Compute phase factors on a regular 3D mesh of q-space samples.

Arguments:
    r (Nx3 numpy array): Atomic coordinates
    f (numpy array): A numpy array of complex atomic scattering factors
    N (numpy array length 3): Number of q-space samples in each of the three 
       dimensions
    q_min (numpy array length 3): Minimum q-space magnitudes in the 3d mesh.  
       These values specify the *center* of the first voxel.
    q_max (numpy array length 3): Naximum q-space magnitudes in the 3d mesh.  
       These values specify the *center* of the voxel.
    context (pyopencl context): Optional, usually via cl.create_some_context()
    queue (pyopencl queue): Optional, usually via cl.CommandQueue(context)
    group_size (int): Optional specification of pyopencl group size (default 64 or
       maximum)

Returns:
    An array of complex scattering amplitudes.  By default this is a normal
       numpy array.  Optionally, this may be an opencl buffer.
    '''

    if context is None:
        context = default_context
    if queue is None:
        queue = default_queue
    if group_size is None:
        group_size = default_group_size

    N = np.array(N, dtype=np.int32)
    q_max = np.array(q_max, dtype=np.float32)
    q_min = np.array(q_min, dtype=np.float32)

    if len(N.shape) == 0:
        N = np.ones(3, dtype=np.int32) * N
    if len(q_max.shape) == 0:
        q_max = np.ones(3, dtype=np.float32) * q_max
    if len(q_min.shape) == 0:
        q_min = np.ones(3, dtype=np.float32) * q_min

    deltaQ = np.array((q_max - q_min) / (N - 1.0), dtype=np.float32)

    n_atoms = np.int32(r.shape[0])
    n_pixels = np.int32(N[0] * N[1] * N[2])

    # Setup buffers.  This is very fast.  However, we are assuming that we can
    # just load all atoms into memory, which might not be possible...
    r_dev = to_device(r, dtype=np.float32, queue=queue)
    f_dev = to_device(f, dtype=np.complex64, queue=queue)
    N = vec4(N, dtype=np.int32)
    deltaQ = vec4(deltaQ, dtype=np.float32)
    q_min = vec4(q_min, dtype=np.float32)
    a_dev = to_device(a, dtype=np.complex64, shape=(n_pixels), queue=queue)

    global_size = np.int(np.ceil(n_pixels / np.float(group_size)) * group_size)

    phase_factor_mesh_cl(queue, (global_size,), (group_size,), r_dev.data,
                         f_dev.data, a_dev.data, n_pixels, n_atoms, N, deltaQ,
                         q_min)

    if a is None:
        return a_dev.get()
    else:
        return a_dev


buffer_mesh_lookup_cl = programs.buffer_mesh_lookup
buffer_mesh_lookup_cl.set_scalar_arg_dtypes(
    [None, None, None, np.int32, None, None, None, None])


def buffer_mesh_lookup(a_map, N, q_min, q_max, q, R=None, a_out=None,
                       context=None, queue=None, group_size=None):
    """
This is supposed to lookup intensities from a 3d mesh of amplitudes.

Arguments:
    a (numpy array): Complex scattering amplitudes (usually generated from the
       function phase_factor_mesh())
    N (int): As defined in phase_factor_mesh()
    q_min (float): As defined in phase_factor_mesh()
    q_max (float): As defined in phase_factor_mesh()
    q (Nx3 numpy array): q-space coordinates at which we want to interpolate
       the complex amplitudes in a_dev
    R (3x3 numpy array): Rotation matrix that will act on the q vectors
    context (pyopencl context): Optional, usually via cl.create_some_context()
    queue (pyopencl queue): Optional, usually via cl.CommandQueue(context)
    group_size (int): Optional specification of pyopencl group size (default 64 or
       maximum)

Returns:
    numpy array of complex amplitudes
    """

    if context is None:
        context = default_context
    if queue is None:
        queue = default_queue
    if group_size is None:
        group_size = default_group_size

    if R is None:
        R = np.eye(3, dtype=np.float32)
    R16 = np.zeros([16], dtype=np.float32)
    R16[0:9] = R.flatten().astype(np.float32)

    N = np.array(N, dtype=np.int32)
    q_max = np.array(q_max, dtype=np.float32)
    q_min = np.array(q_min, dtype=np.float32)

    if len(N.shape) == 0:
        N = np.ones(3, dtype=np.int32) * N
    if len(q_max.shape) == 0:
        q_max = np.ones(3, dtype=np.float32) * q_max
    if len(q_min.shape) == 0:
        q_min = np.ones(3, dtype=np.float32) * q_min

    deltaQ = np.array((q_max - q_min) / (N - 1.0), dtype=np.float32)

    n_pixels = np.int32(q.shape[0])

    a_map_dev = to_device(a_map, dtype=np.complex64, queue=queue)
    q_dev = to_device(q, dtype=np.float32, queue=queue)
    N = vec4(N, dtype=np.int32)
    deltaQ = vec4(deltaQ, dtype=np.float32)
    q_min = vec4(q_min, dtype=np.float32)
    a_out_dev = to_device(
        a_out, dtype=np.complex64, shape=(n_pixels), queue=queue)

    global_size = np.int(np.ceil(n_pixels / np.float(group_size)) * group_size)

    buffer_mesh_lookup_cl(queue, (global_size,), (group_size,), a_map_dev.data,
                          q_dev.data, a_out_dev.data, n_pixels, N, deltaQ,
                          q_min, R16)

    if a_out is None:
        return a_out_dev.get()
    else:
        return None


class ClCore(object):
    
    def __init__(self,context=None,queue=None,group_size=None):
        
        self.context = None
        self.queue = None
        self.group_size = None
        self.programs = None
        
        # Abstract real and complex types to allow for double/single in the
        # future.
        self.int_t = np.int32
        self.real_t = np.float32
        self.complex_t = np.complex64
        
        # Setup the context
        if context is None:
            self.context = cl.create_some_context()
        else:
            self.context = context
        
        # Setup the queue
        if queue is None:
            self.queue = cl.CommandQueue(self.context)
        else:
            self.queue = queue
        
        # Setup the group size.
        # If the environment variable BORNAGAIN_CL_GROUPSIZE is set then use
        # that value.
        # If the group size exceeds the max allowed group size, then make it
        # smaller (but print warning)
        if os.environ.get('BORNAGAIN_CL_GROUPSIZE') is not None:
            group_size = np.int(os.environ.get('BORNAGAIN_CL_GROUPSIZE'))
        if group_size is None:
            group_size = 32
        max_group_size = self.queue.device.max_work_group_size
        if group_size > max_group_size:
            sys.stderr.write('Changing group size from %d to %d.\n'
                     'Set BORNAGAIN_CL_GROUPSIZE=%d to avoid this error.\n' 
                     % (group_size, max_group_size, max_group_size))
            group_size = max_group_size
        self.group_size = group_size
        
        # Build the programs
        self.programs = cl.Program(self.context, open(clcore_file).read()
                                   ).build(
                        options=['-D', 'GROUP_SIZE=%d' % self.group_size])

        # Configure the python interface to the cl programs
        self.get_group_size_cl = self.programs.get_group_size
        self.get_group_size_cl.set_scalar_arg_dtypes([None])
        
        self.phase_factor_qrf_cl = self.programs.phase_factor_qrf
        self.phase_factor_qrf_cl.set_scalar_arg_dtypes(
                    [None, None, None, None, None, self.int_t, self.int_t])
        
        self.phase_factor_pad_cl = self.programs.phase_factor_pad
        self.phase_factor_pad_cl.set_scalar_arg_dtypes(
             [None, None, None, None, self.int_t, self.int_t, self.int_t, self.int_t,
              self.real_t, None, None, None, None])
        
    def vec4(self, x, dtype=None):
        """
Evdidently pyopencl does not deal with 3-vectors very well, so we use
4-vectors and pad with a zero at the end.
    
Arguments:
    x: array
    dtype: Optional np.dtype (default np.float32)
    
Returns:
    numpy array of length 4
        """
        
        if dtype is None:
            dtype = self.real_t
        return np.array([x[0], x[1], x[2], 0.0], dtype=dtype)
 

    def to_device(self, array=None, shape=None, dtype=None):
        """
This is a thin wrapper for pyopencl.array.to_device().  It will convert a numpy 
array into a pyopencl.array and send it to the device memory.  So far this only
deals with float and comlex arrays, and it should figure out which type it is.

Arguments:
    array (numpy/cl array; float/complex type): Input array.
    shape (tuple): Optionally specify the shape of the desired array.  This is 
                    ignored if array is not None. 
    dtype (np.dtype): Specify the desired type in opencl.  The two types that 
                       are useful here are np.float32 and np.complex64

Returns:
    pyopencl array
        """

        if isinstance(array, cl.array.Array):
            return array

        if array is None:
            array = np.zeros(shape, dtype=dtype)

        if dtype is None:
            if np.iscomplexobj(array):
                dtype = self.complex_t
            else:
                dtype = self.real_t

        return cl.array.to_device(self.queue, 
                                  np.ascontiguousarray(array.astype(dtype)))


    def get_group_size(self):
    
        group_size_dev = self.to_device(np.zeros((1)), dtype=self.int_t)
        self.get_group_size_cl(self.queue, (self.group_size,), 
                               (self.group_size,), group_size_dev.data)
        
        
        return group_size_dev.get()[0]

    def phase_factor_qrf(self, q, r, f, R=None, a=None):
        '''
Calculate diffraction amplitudes: sum over f_n*exp(-iq.r_n)

Arguments:
    q (numpy/cl float array [N,3]): Scattering vectors (2\pi/\lambda).
    r (numpy/cl float array [M,3]): Atomic coordinates.
    f (numpy/cl complex array [M]): Complex scattering factors.
    R (numpy array [3,3]): Rotation matrix acting on q vectors.
    a (cl complex array [N]): Optional container for complex scattering 
      amplitudes.
    context (pyopencl context): Optional pyopencl context (e.g. use 
      cl.create_some_context() to create one).
    queue (pyopencl context): Optional pyopencl queue (e.g use 
      cl.CommandQueue(context) to create one).
    group_size (int): Optional pyopencl group size.

Returns:
    (numpy/cl complex array [N]): Diffraction amplitudes.  Will be a cl array 
      if there are input cl arrays.
        '''

        if R is None:
            R = np.eye(3, dtype=self.real_t)
        R16 = np.zeros([16], dtype=self.real_t)
        R16[0:9] = R.flatten().astype(self.real_t)
    
        n_pixels = self.int_t(q.shape[0])
        n_atoms = self.int_t(r.shape[0])
        q_dev = self.to_device(q, dtype=self.real_t)
        r_dev = self.to_device(r, dtype=self.real_t)
        f_dev = self.to_device(f, dtype=self.complex_t)
        a_dev = self.to_device(a, dtype=self.complex_t, shape=(n_pixels))
    
        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) 
                             * self.group_size)
    
        self.phase_factor_qrf_cl(self.queue, (global_size,), 
                                 (self.group_size,), q_dev.data, r_dev.data, 
                                 f_dev.data, R16, a_dev.data, n_atoms,
                                 n_pixels)
    
        if a is None:
            return a_dev.get()
        else:
            return a_dev
        
    def phase_factor_pad(self, r, f, T, F, S, B, nF, nS, w, R=None,
                         a=None):
        '''
This should simulate detector panels.

Arguments:
    r: An Nx3 numpy array with atomic coordinates (meters)
    f: A numpy array with complex scattering factors
    T: A 1x3 numpy array with vector components pointing from sample to
       the center of the first pixel in memory
    F: A 1x3 numpy array containing the basis vector components pointing
        in the direction corresponding to contiguous pixels in memory
        ("fast scan").
    S: A 1x3 numpy array containing the basis vector components pointing
        in the direction corresponding to non-contiguous pixels in
        memory ("slow scan").
    B: A 1x3 numpy array with unit-vector components corresponding to the
        incident x-ray beam direction
    nF: Number of fast-scan pixels (corresponding to F vector) in the
        detector panel
    nS: Number of slow-scan pixels (corresponding to S vector) in the
        detector panel
    w: The photon wavelength in meters
    R: Optional numpy array [3x3] specifying rotation of q vectors
    a: Optional output complex scattering amplitude cl array

Returns:
    A: A numpy array of length nF*nS containing complex scattering
amplitudes
        '''
    
        if R is None:
            R = np.eye(3, dtype=self.real_t)
        R16 = np.zeros([16], dtype=self.real_t)
        R16[0:9] = R.flatten().astype(self.real_t)
    
        nF = self.int_t(nF)
        nS = self.int_t(nS)
        n_pixels = self.int_t(nF * nS)
        n_atoms = self.int_t(r.shape[0])
        r_dev = self.to_device(r, dtype=self.real_t)
        f_dev = self.to_device(f, dtype=self.complex_t)
        T = self.vec4(T)
        F = self.vec4(F)
        S = self.vec4(S)
        B = self.vec4(B)
        a_dev = self.to_device(a, dtype=self.complex_t, shape=(n_pixels))
    
        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) * 
                             self.group_size)
    
        self.phase_factor_pad_cl(self.queue, (global_size,), 
                                 (self.group_size,), r_dev.data,
                            f_dev.data, R16, a_dev.data, n_pixels, n_atoms, 
                            nF, nS, w, T, F, S, B)
    
        if a is None:
            return a_dev.get()
        else:
            return a_dev







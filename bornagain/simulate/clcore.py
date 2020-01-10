r"""
This module contains some core functions that are useful for simulating diffraction on GPU devices.  Look to the
bornagain documentation to gain an understaning of how this module is meant to be used.

To get some information on compute devices (CPU/GPU) you can run the function clcore.help()

Some environment variables that affect the behaviour of this module:

* BORNAGAIN_CL_GROUPSIZE : This sets the default groupsize.
* PYOPENCL_CTX: This sets the device and platform automatically.

Using the above variables allows you to run the same code on different devices.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import sys

import numpy as np
import pkg_resources
import pyopencl as cl
import pyopencl.array

import bornagain as ba
from bornagain.utils import depreciate
from bornagain.simulate import refdata

cl_array = cl.array.Array

clcore_file = pkg_resources.resource_filename('bornagain.simulate', 'clcore.cpp')


def create_some_gpu_context():

    r"""

    Since cl.create_some_context() sometimes forces a CPU on macs, this function will attempt to use a GPU
    context if possible.

    Returns: opencl context

    """

    try:
        platform = cl.get_platforms()
        devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        context = cl.Context(devices=devices)
    except:
        context = cl.create_some_context()

    return context


class ClCore(object):

    r"""

    A container for the elementary building blocks that GPU diffraction simulations are composed of.

    An instance of this class will initialize an opencl context and help maintain consistency in the
    device queue that compute jobs are sent to, along with consistency in the precision (double/single)
    when memory moves between CPU and GPU memory.

    """

    def __init__(self, context=None, queue=None, group_size=32, double_precision=False):

        r"""

        An instance of this class will attempt to help you manage an opencl context and command queue.
        You may choose the precision that you desire from the beginning, and this will be taken care of
        so that you don't need to think about it when you move arrays between CPU and GPU memory.

        An opencl context and queue will be created if you do not provide them.  This is the most common
        mode of operation.

        You may choose the group size, which is the number of compute units that have shared local
        memory.  The environment variable BORNAGAIN_CL_GROUPSIZE can be used to set the default group
        size for a given machine if you want your code to be the same on multiple different machines.

        The raw opencl kernel code will be compiled when an instance of this class is created.

        Arguments:
            context: An opencl context
            queue: An opencl queue
            group_size (int): The desired opencl group size (most common is 32, and this is default).
            double_precision (bool): True if double-precision is desired
        """

        self.group_size = None
        self.programs = None
        self.double_precision = double_precision

        # Setup the context
        if context is None:
            self.context = create_some_gpu_context()
        else:
            self.context = context

        # Setup the queue
        if queue is None:
            self.queue = cl.CommandQueue(self.context)
        else:
            self.queue = queue

        # Abstract real and complex types to allow for double/single
        self._setup_precision(double_precision)

        # Setup the group size.
        self.set_groupsize(group_size)

        # setup the programs
        self._build_opencl_programs()

    def set_groupsize(self, group_size):
        r"""
        If the environment variable BORNAGAIN_CL_GROUPSIZE is set then use
        that value.

        If the group size exceeds the max allowed group size, then make it
        smaller (but print warning)
        """

        if os.environ.get('BORNAGAIN_CL_GROUPSIZE') is not None:
            group_size = np.int(os.environ.get('BORNAGAIN_CL_GROUPSIZE'))
        max_group_size = self.queue.device.max_work_group_size
        if self.double_precision:
            max_group_size = int(max_group_size / 2)
        if group_size > max_group_size:
            # FIXME: messages of the type below should be printed only in debug mode.
            # sys.stderr.write('Changing group size from %d to %d.\n'
            #                  'Set BORNAGAIN_CL_GROUPSIZE=%d to avoid this error.\n'
            #                  % (group_size, max_group_size, max_group_size))
            group_size = max_group_size
        self.group_size = group_size

    def _double_precision_is_available(self):
        if 'cl_khr_fp64' not in self.queue.device.extensions.split():
            return False
        # TODO: fix stupid errors to do with Apple's CL double implementation?  Why doesn't double work on apple?
        if self.queue.device.platform.name == 'Apple':
            return False
        return True

    def _setup_precision(self, dbl):
        if not dbl:
            self._use_float()
            self.double_precision = False
        if dbl:
            if self._double_precision_is_available():
                sys.stderr.write('Attempting to use double precision.\n')
                self._use_double()
                self.double_precision = True
            else:
                sys.stderr.write('Double precision not supported on\n%s'
                                 '\nFallback to single precision\n'
                                 % self.queue.device.name)
                self.double_precision = False
                self._use_float()

    def _use_double(self):
        # TODO: Decide if integers should be made double also.  As of now, they are all single precision.
        self.int_t = np.int32
        self.real_t = np.float64
        self.complex_t = np.complex128

    def _use_float(self):
        self.int_t = np.int32
        self.real_t = np.float32
        self.complex_t = np.complex64

    # def _load_programs(self):
    #     self._build_opencl_programs()

    def _build_opencl_programs(self):
        clcore_file = pkg_resources.resource_filename('bornagain.simulate', 'clcore.cpp')
        kern_str = open(clcore_file).read()
        build_opts = []
        if self.double_precision:
            build_opts.append('-D')
            build_opts.append('CONFIG_USE_DOUBLE=1')
        build_opts.append('-D')
        build_opts.append('GROUP_SIZE=%d' % (self.group_size))
        self.programs = cl.Program(self.context, kern_str).build(options=build_opts)

    def vec4(self, x, dtype=None):

        r"""
        Evdidently pyopencl does not deal with 3-vectors very well, so we use
        4-vectors and pad with a zero at the end.

        From Derek: I tested this at one point and found no difference... maybe newer pyopenCL is better..

        This just does a trivial operation:
        return np.array([x.flat[0], x.flat[1], x.flat[2], 0.0], dtype=dtype)

        Arguments:
            x np.ndarray:

            dtype np.dtype: Examples: np.complex, np.double

        Returns:
            - numpy array of length 4
        """

        if dtype is None:
            dtype = self.real_t
        return np.array([x.flat[0], x.flat[1], x.flat[2], 0.0], dtype=dtype)

    def vec16(self, R, dtype=None):

        r"""
        The best way to pass in a rotation matrix is as a float16.  This is a helper function for
        preparing a numpy array so that it can be passed in as a float16.

        From Derek: I had tested this and found no difference

        See the vec4 function documentation also.

        Arguments:
            R numpy.ndarray: input array

            dtype numpy.dtype: default is np.float32

        Returns:
            - numpy array of length 16
        """

        if dtype is None:
            dtype = self.real_t
        R16 = np.zeros([16], dtype=dtype)
        R16[0:9] = R.flatten().astype(dtype)
        return R16

    def to_device(self, array=None, shape=None, dtype=None):

        r"""
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

        if isinstance(array, cl_array):
            return array

        if array is None:
            array = np.zeros(shape, dtype=dtype)

        if dtype is None:
            if np.iscomplexobj(array):
                dtype = self.complex_t
            else:
                dtype = self.real_t

        return cl.array.to_device(self.queue, np.ascontiguousarray(array.astype(dtype)))

    def get_group_size(self):

        r"""
        retrieve the currently set group_size
        """

        if not hasattr(self, 'get_group_size_cl'):
            self.get_group_size_cl = self.programs.get_group_size
            self.get_group_size_cl.set_scalar_arg_dtypes([None])

        group_size_dev = self.to_device(np.zeros(1), dtype=self.int_t)
        self.get_group_size_cl(self.queue, (self.group_size,), (self.group_size,), group_size_dev.data)

        return group_size_dev.get()[0]

    def _next_multiple_groupsize(self, N):

        if N % self.group_size > 0:
            return self.int_t(self.group_size - N % self.group_size)
        else:
            return 0

    def rotate_translate_vectors(self, rot, trans, vec_in, vec_out=None):

        r"""

        Apply rotation followed by translation on GPU.

        Arguments:
            rot: rotation matrix
            trans: translation vector
            v_in (Nx3 array): input vectors
            v_out (Nx3 array): output vectors

        Returns: output vectors

        """

        if not hasattr(self, 'rotate_translate_vectors_cl'):
            self.rotate_translate_vectors_cl = self.programs.rotate_translate_vectors
            self.rotate_translate_vectors_cl.set_scalar_arg_dtypes([None, None, None, None, self.int_t])

        n_vecs = self.int_t(vec_in.shape[0])

        if vec_out is None:
            vec_out_dev = self.to_device(shape=vec_in.shape, dtype=self.real_t)
        else:
            vec_out_dev = self.to_device(vec_out, dtype=self.real_t)

        rot_dev = self.vec16(rot)
        trans_dev = self.vec4(trans)

        vec_in_dev = self.to_device(vec_in, dtype=self.real_t)

        global_size = np.int(np.ceil(n_vecs / np.float(self.group_size)) * self.group_size)

        self.rotate_translate_vectors_cl(self.queue, (global_size,), (self.group_size,), rot_dev, trans_dev,
                                vec_in_dev.data, vec_out_dev.data, n_vecs)

        if vec_out is None:
            return vec_out_dev.get()
        else:
            return None

    def test_rotate_vec(self, rot, trans, vec):

        r"""
        Rotate a single vector.  CPU arrays in, CPU array out. This is just for testing the consistency of memory
        allocation.
        """

        if not hasattr(self, 'test_rotate_vec_cl'):
            self.test_rotate_vec_cl = self.programs.test_rotate_vec
            self.test_rotate_vec_cl.set_scalar_arg_dtypes([None, None, None, None])

        rot = self.vec16(rot)
        vec = self.vec4(vec)
        trans = self.vec4(trans)
        vec_out = vec.copy()
        vec_out_dev = self.to_device(vec_out, dtype=self.real_t)
        n = 1

        global_size = np.int(np.ceil(n / np.float(self.group_size)) * self.group_size)

        self.test_rotate_vec_cl(self.queue, (global_size,), (self.group_size,), rot, trans, vec, vec_out_dev.data)

        return vec_out_dev.get()[0:3]

    def test_simple_sum(self, vec):
        r""" For testing -- appears in the pytest files """

        if not hasattr(self, 'test_rotate_vec_cl'):
            self.test_simple_sum_cl = self.programs.test_simple_sum
            self.test_simple_sum_cl.set_scalar_arg_dtypes([None, None, self.int_t])

        n = self.int_t(len(vec))
        out = np.array((1,), self.real_t)
        vec_dev = self.to_device(vec, dtype=self.real_t)
        out_dev = self.to_device(out, dtype=self.real_t)
        global_size = self.group_size
        self.test_simple_sum_cl(self.queue, (global_size,), (self.group_size,), vec_dev.data, out_dev.data, n)
        return out_dev.get()[0]

    def mod_squared_complex_to_real(self, A, I):

        r"""
        Compute the real-valued modulus square of complex numbers.  Good example of a function that
        shouldn't exist, but I needed to add it here because the pyopencl.array.Array class fails to
        do this operation correctly on some computers.
        """

        if not hasattr(self, 'mod_squared_complex_to_real_cl'):
            self.mod_squared_complex_to_real_cl = self.programs.mod_squared_complex_to_real
            self.mod_squared_complex_to_real_cl.set_scalar_arg_dtypes([None, None, self.int_t])

        A_dev = self.to_device(A, dtype=self.complex_t)
        I_dev = self.to_device(I, dtype=self.real_t)
        n = self.int_t(np.prod(A.shape))

        global_size = np.int(np.ceil(n / np.float(self.group_size)) * self.group_size)

        self.mod_squared_complex_to_real_cl(self.queue, (global_size,), (self.group_size,), A_dev.data, I_dev.data, n)

    def phase_factor_qrf_chunk_r(self, q, r, f=None, R=None, U=None, a=None, add=False, n_chunks=1):

        r"""

        This needs to be tested, made for really big atom-vectors (e.g. a virus or rhibosome)

        Calculate diffraction amplitudes: sum over f_n*exp(-iq.r_n)

        Arguments:
            q (numpy/cl float array [N,3]): Scattering vectors (2\pi/\lambda).
            r (numpy/cl float array [M,3]): Atomic coordinates.
            f (numpy/cl complex array [M]): Complex scattering factors.
            a (cl complex array [N]): Optional container for complex scattering amplitudes.
            R (numpy array [3,3]): Rotation matrix acting on atom vectors.
                (we quietly transpose R and let it operate on q-vectors for speedups)
            add (bool): Set to true if you want to add to the input amplitude array "a".  Else it is overwritten.
            n_chunks (int): Number of chunks to split up atoms.


        Returns:
            (numpy/cl complex array [N]): Diffraction amplitudes.  Will be a cl array
              if there are input cl arrays.
        """

        # We must do this because pyopencl Array objects do not allow array slicing.
        if type(r) is cl_array or type(f) is cl_array:
            raise ValueError('phase_factor_qrf_chunk_r requires that r and f are numpy arrays.')

        add = self.int_t(add)

        if f is None:
            f = np.ones(r.shape[0])

        n_pixels = self.int_t(q.shape[0])
        n_atoms = self.int_t(r.shape[0])
        q_dev = self.to_device(q, dtype=self.real_t)
        a_dev = self.to_device(a, dtype=self.complex_t, shape=(n_pixels,))

        r_split = np.array_split(np.arange(n_atoms), n_chunks)
        for i in range(0, len(r_split)):
            r_rng = r_split[i]
            r_chunk = r[r_rng[0]:(r_rng[-1]+1), :]
            f_chunk = f[r_rng[0]:(r_rng[-1]+1)]
            if i > 0:
                add = self.int_t(1)
            self.phase_factor_qrf(q_dev, r_chunk, f_chunk, R, U, a_dev, add)

        if a is None:
            return a_dev.get()
        else:
            return a_dev

    def phase_factor_qrf(self, q, r, f=None, R=None, U=None, a=None, add=False):

        r"""
        Calculate diffraction amplitudes: sum over f_n*exp(-iq.r_n)

        Arguments:
            q (numpy/cl float array [N,3]): Scattering vectors (2\pi/\lambda).
            r (numpy/cl float array [M,3]): Atomic coordinates.
            R (numpy array [3,3]): Rotation matrix acting on atom vectors.
                (we quietly transpose R and let it operate on q-vectors for speedups)
            f (numpy/cl complex array [M]): Complex scattering factors.
            a (cl complex array [N]): Optional container for complex scattering
              amplitudes.

        Returns:
            (numpy/cl complex array [N]): Diffraction amplitudes.  Will be a cl array
              if there are input cl arrays.
        """

        if not hasattr(self, 'phase_factor_qrf_cl'):
            self.phase_factor_qrf_cl = self.programs.phase_factor_qrf
            self.phase_factor_qrf_cl.set_scalar_arg_dtypes(
                [None, None, None, None, None, None, self.int_t, self.int_t, self.int_t])

        if R is None:
            R = np.eye(3)

        if U is None:
            U = np.zeros(3, dtype=self.real_t)
        U = np.dot(R.T, U)

        if add:
            add = 1
        else:
            add = 0

        if f is None:
            f = np.ones(r.shape[0])

        n_pixels = self.int_t(q.shape[0])
        n_atoms = self.int_t(r.shape[0])
        q_dev = self.to_device(q, dtype=self.real_t)
        r_dev = self.to_device(r, dtype=self.real_t)
        f_dev = self.to_device(f, dtype=self.complex_t)
        a_dev = self.to_device(a, dtype=self.complex_t, shape=(n_pixels))
        R = self.vec16(R.T, dtype=self.real_t)
        U = self.vec4(U, dtype=self.real_t)

        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) * self.group_size)

        self.phase_factor_qrf_cl(self.queue, (global_size,),
                                 (self.group_size,), q_dev.data, r_dev.data,
                                 f_dev.data, R, U, a_dev.data, n_atoms,
                                 n_pixels, add)
        self.queue.finish()

        if a is None:
            return a_dev.get()
        else:
            return a_dev

    def phase_factor_pad(self, r, f=None, T=None, F=None, S=None, B=None, nF=None, nS=None, w=None, R=None, U=None, a=None,
                         add=False, beam=None, pad=None):

        r"""
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
            R: Optional numpy array [3x3] specifying rotation of atom vectors
                (we quietly transpose R and let it operate on q-vectors for speedups)
            a: Optional output complex scattering amplitude cl array

        Returns:
            A: A numpy array of length nF*nS containing complex scattering
        amplitudes
        """

        if not hasattr(self, 'phase_factor_pad_cl'):
            self.phase_factor_pad_cl = self.programs.phase_factor_pad
            self.phase_factor_pad_cl.set_scalar_arg_dtypes(
                [None, None, None, None, None, self.int_t, self.int_t, self.int_t, self.int_t, self.real_t,
                 None, None, None, None, self.int_t])

        n_atoms = r.shape[0]
        if f is None:
            f = np.ones(n_atoms, dtype=self.complex_t)

        if R is None:
            R = np.eye(3)

        if U is None:
            U = np.zeros(3, dtype=self.real_t)
        U = np.dot(R.T, U)

        if add:
            add = 1
        else:
            add = 0

        if beam is None and pad is None:
            w = self.real_t(w)
            nF = self.int_t(nF)
            nS = self.int_t(nS)
            n_pixels = self.int_t(nF * nS)
            n_atoms = self.int_t(n_atoms)
            r_dev = self.to_device(r, dtype=self.real_t)
            f_dev = self.to_device(f, dtype=self.complex_t)
            R = self.vec16(R.T, dtype=self.real_t)
            T = self.vec4(T, dtype=self.real_t)
            F = self.vec4(F, dtype=self.real_t)
            S = self.vec4(S, dtype=self.real_t)
            B = self.vec4(B, dtype=self.real_t)
            U = self.vec4(U, dtype=self.real_t)
        elif beam is not None and pad is not None:
            w = self.real_t(beam.wavelength)
            nF = self.int_t(pad.n_fs)
            nS = self.int_t(pad.n_ss)
            n_pixels = self.int_t(pad.n_pixels)
            n_atoms = self.int_t(n_atoms)
            r_dev = self.to_device(r, dtype=self.real_t)
            f_dev = self.to_device(f, dtype=self.complex_t)
            R = self.vec16(R.T, dtype=self.real_t)
            T = self.vec4(pad.t_vec, dtype=self.real_t)
            F = self.vec4(pad.fs_vec, dtype=self.real_t)
            S = self.vec4(pad.ss_vec, dtype=self.real_t)
            B = self.vec4(beam.beam_vec, dtype=self.real_t)
            U = self.vec4(U, dtype=self.real_t)
        else:
            raise ValueError('Ether beam and pad must be provided, or provide all parameters separately')

        a_dev = self.to_device(a, dtype=self.complex_t, shape=(n_pixels))

        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) * self.group_size)

        self.phase_factor_pad_cl(self.queue, (global_size,),
                                 (self.group_size,), r_dev.data,
                                 f_dev.data, R, U, a_dev.data, n_pixels, n_atoms,
                                 nF, nS, w, T, F, S, B, add)
        self.queue.finish()

        if a is None:
            return a_dev.get()
        else:
            return a_dev

    def phase_factor_mesh(self, r, f, N=None, q_min=None, q_max=None, dq=None, a=None, R=None, U=None, add=False,
                          density_map=None):

        r"""
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
            dq (numpy array of length 3): For legacy reasons, you can specify dq instead of N.
                                          Note that the relation between dq and N is dq = (q_max-q_min)/(N-1).
            a (clArray): device buffer, if available
            R (3x3 array): Rotation matrix, to be applied to r vectors
            add (bool): If True, add to device buffer a, else overwrite the buffer.
            density_map (an instance of ):

        Returns:
            An array of complex scattering amplitudes.  By default this is a normal
               numpy array.  Optionally, this may be an opencl buffer.
        """

        if not hasattr(self, 'phase_factor_mesh_cl'):
            self.phase_factor_mesh_cl = self.programs.phase_factor_mesh
            self.phase_factor_mesh_cl.set_scalar_arg_dtypes(
                [None, None, None, self.int_t, self.int_t, None, None, None, None, None, self.int_t])

        if add:
            add = 1
        else:
            add = 0
        if R is None:
            R = np.eye(3)

        if U is None:
            U = np.zeros(3, dtype=self.real_t)
        U = np.dot(R.T, U)

        R = self.vec16(R.T, dtype=self.real_t)
        U = self.vec4(U, dtype=self.real_t)

        if density_map is None:
            N = np.array(N, dtype=self.int_t)
            if q_max is None:
                q_max = q_min + dq*(N-1)
            q_max = np.array(q_max, dtype=self.real_t)
            q_min = np.array(q_min, dtype=self.real_t)
        else:
            N = np.array(density_map.shape, dtype=self.int_t)
            q_min = np.array(density_map.limits[:, 0], dtype=self.real_t)
            q_max = np.array(density_map.limits[:, 1], dtype=self.real_t)

        if len(N.shape) == 0:
            N = np.ones(3, dtype=self.int_t) * N
        if len(q_max.shape) == 0:
            q_max = np.ones(3, dtype=self.real_t) * q_max
        if len(q_min.shape) == 0:
            q_min = np.ones(3, dtype=self.real_t) * q_min

        deltaQ = np.array((q_max - q_min) / (N - 1.0), dtype=self.real_t)

        n_atoms = self.int_t(r.shape[0])
        n_pixels = self.int_t(N[0] * N[1] * N[2])

        # Setup buffers.  This is very fast.  However, we are assuming that we can
        # just load all atoms into memory, which might not be possible...
        r_dev = self.to_device(r, dtype=self.real_t)
        f_dev = self.to_device(f, dtype=self.complex_t)
        N = self.vec4(N, dtype=self.int_t)
        deltaQ = self.vec4(deltaQ, dtype=self.real_t)
        q_min = self.vec4(q_min, dtype=self.real_t)
        a_dev = self.to_device(a, dtype=self.complex_t, shape=(n_pixels))

        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size))
                             * self.group_size)

        self.phase_factor_mesh_cl(self.queue, (global_size,),
                                  (self.group_size,), r_dev.data, f_dev.data,
                                  a_dev.data, n_pixels, n_atoms, N, deltaQ,
                                  q_min, R, U, add)
        self.queue.finish()

        if a is None:
            return a_dev.get()
        else:
            return a_dev

    def buffer_mesh_lookup(self, *args, **kwargs):

        depreciate("buffer_mesh_lookup has been renamed to mesh_interpolation")

        return self.mesh_interpolation(*args, **kwargs)

    def mesh_interpolation(self, a_map, q, N=None, q_min=None, q_max=None, dq=None, R=None, U=None, a=None,
                           density_map=None, add=False):

        r"""
        This is supposed to lookup intensities from a 3d mesh of amplitudes.

        Arguments:
            a_map (numpy array): Complex scattering amplitudes (usually generated from
               the function phase_factor_mesh()).  Can also be real amplitudes.
            N (int): As defined in phase_factor_mesh()
            q_min (float): As defined in phase_factor_mesh()
            q_max (float): As defined in phase_factor_mesh()
            q (Nx3 numpy array): q-space coordinates at which we want to interpolate
               the complex amplitudes in a_dev
            R (3x3 numpy array): Rotation matrix that will act on the atom vectors
                (we quietly transpose R and let it operate on q-vectors for speedups)
            a: (clarray) The output array (optional)

        Returns:
            numpy array of complex amplitudes
        """

        if not hasattr(self, 'mesh_interpolation_cl'):
            arg_types = [None, None, None, self.int_t, None, None, None, None, None, self.int_t, self.int_t]
            self.mesh_interpolation_cl = self.programs.mesh_interpolation
            self.mesh_interpolation_cl.set_scalar_arg_dtypes(arg_types)
            self.mesh_interpolation_real_cl = self.programs.mesh_interpolation_real
            self.mesh_interpolation_real_cl.set_scalar_arg_dtypes(arg_types)

        if add is True:
            add = self.int_t(1)
        else:
            add = self.int_t(0)

        if R is None:
            R = np.eye(3)

        if U is None:
            do_translate = self.int_t(0)
            U = np.zeros(3, dtype=self.real_t)
        else:
            do_translate = self.int_t(1)
            U = np.dot(R.T, U)

        R = self.vec16(R.T, dtype=self.real_t)
        U = self.vec4(U, dtype=self.real_t)

        if density_map is None:
            N = np.array(N, dtype=self.int_t)
            if q_max is None:
                q_max = q_min + dq*(N-1)
            q_max = np.array(q_max, dtype=self.real_t)
            q_min = np.array(q_min, dtype=self.real_t)
        else:
            N = np.array(density_map.shape, dtype=self.int_t)
            q_min = np.array(density_map.limits[:, 0], dtype=self.real_t)
            q_max = np.array(density_map.limits[:, 1], dtype=self.real_t)

        if len(N.shape) == 0:
            N = (np.ones(3) * N).astype(self.int_t)
        if len(q_max.shape) == 0:
            q_max = self.real_t(np.ones(3) * q_max)
        if len(q_min.shape) == 0:
            q_min = self.real_t(np.ones(3) * q_min)

        dq = np.array((q_max - q_min) / (N - 1.0), dtype=self.real_t)

        n_pixels = self.int_t(q.shape[0])

        if a_map.dtype == self.complex_t:
            a_map_dev = self.to_device(a_map, dtype=self.complex_t)
            q_dev = self.to_device(q, dtype=self.real_t)
            N = self.vec4(N, dtype=self.int_t)
            dq = self.vec4(dq, dtype=self.real_t)
            q_min = self.vec4(q_min, dtype=self.real_t)
            a_out_dev = self.to_device(a, dtype=self.complex_t, shape=(n_pixels,))

            global_size = np.int(np.ceil(n_pixels / np.float(self.group_size))
                                 * self.group_size)

            self.mesh_interpolation_cl(self.queue, (global_size,), (self.group_size,),
                                       a_map_dev.data, q_dev.data, a_out_dev.data,
                                       n_pixels, N, dq, q_min, R, U, do_translate, add)
        elif a_map.dtype == self.real_t:
            a_map_dev = self.to_device(a_map, dtype=self.real_t)
            q_dev = self.to_device(q, dtype=self.real_t)
            N = self.vec4(N, dtype=self.int_t)
            dq = self.vec4(dq, dtype=self.real_t)
            q_min = self.vec4(q_min, dtype=self.real_t)
            a_out_dev = self.to_device(a, dtype=self.real_t, shape=(n_pixels,))

            global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) * self.group_size)

            self.mesh_interpolation_real_cl(self.queue, (global_size,), (self.group_size,), a_map_dev.data, q_dev.data,
                                            a_out_dev.data, n_pixels, N, dq, q_min, R, U, do_translate, add)

        self.queue.finish()

        if a is None:
            return a_out_dev.get()
        else:
            return a_out_dev

    # def mesh_insertion(self, densities, weights, vecs, vals, shape, corner, deltas, rot=None):
    #
    #     r"""
    #     Undocomented
    #     """
    #     # TODO: documentation
    #
    #     if not hasattr(self, 'mesh_insertion_cl'):
    #         arg_types = [None, None, None, None, self.int_t, None, None, None, None]
    #         self.mesh_insertion_cl = self.programs.mesh_insertion
    #         self.mesh_insertion_cl.set_scalar_arg_dtypes(arg_types)
    #         self.mesh_insertion_real_cl = self.programs.mesh_insertion_real
    #         self.mesh_insertion_real_cl.set_scalar_arg_dtypes(arg_types)
    #
    #     if rot is None:
    #         rot = np.eye(3)
    #     rot = self.vec16(rot.T, dtype=self.real_t)
    #     shape = np.array(shape, dtype=np.int)
    #     corner = np.array(corner, dtype=self.real_t)
    #     deltas = np.array(deltas, dtype=self.real_t)
    #     vecs_gpu = self.to_device(vecs, dtype=self.real_t)
    #     n_pixels = self.int_t(vecs.shape[0])
    #
    #     global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) * self.group_size)
    #
    #
    #     if densities.dtype == self.complex_t:
    #         assert vals.dtype == self.complex_t
    #         vals_gpu = self.to_device(vals, dtype=self.complex_t)
    #         self.mesh_insertion_cl(self.queue, (global_size,), (self.group_size,), densities.data, weights.data,
    #                                vecs_gpu.data, vals_gpu.data, n_pixels, shape, deltas, corner, rot)
    #     elif densities.dtype == self.real_t:
    #         assert vals.dtype == self.real_t
    #         vals_gpu = self.to_device(vals, dtype=self.real_t)
    #         print(rot,vals_gpu,vecs_gpu,global_size)
    #         self.mesh_insertion_real_cl(self.queue, (global_size,), (self.group_size,), densities.data, weights.data,
    #                                vecs_gpu.data, vals_gpu.data, n_pixels, shape, deltas, corner, rot)
    #         print(densities, weights)
    #     self.queue.finish()

    def lattice_transform_intensities_pad(self, abc, N, T, F, S, B, nF, nS, w, R=None, I=None, add=False):
        r"""
        Calculate crystal lattice transform intensities for a pixel-array detector.  This is the usual transform for
        an idealized parallelepiped crystal (usually not very realistic...).

        Arguments:
            abc (numpy array) : A 3x3 array containing real-space basis vectors.  Vectors are contiguous in memory.
            N (numpy array)   : An array containing number of unit cells along each of three axes.
            T (numpy array)   : Translation to center of corner pixel.
            F (numpy array)   : Fast-scan basis vector.
            S (numpy array)   : Slow-scan basis vector.
            B (numpy array)   : Incident beam vector.
            nF (int)          : Number of fast-scan pixels.
            nS (int)          : Number of slow-scan pixels.
            w (float)         : Wavelength.
            R (numpy array)   : Rotation matrix acting on atom vectors.
                (we quietly transpose R and let it operate on q-vectors for speedups)
            I (:class:pyopencl.array.Array) : OpenCL device array containing intensities.
            add (bool)        : If true, the function will add to the input I buffer, else the buffer is overwritten.

        Returns:
            If I == None, then the output is a numpy array.  Otherwise, it is an opencl array.
        """

        if not hasattr(self, 'lattice_transform_intensities_pad_cl'):
            self.lattice_transform_intensities_pad_cl = self.programs.lattice_transform_intensities_pad
            self.lattice_transform_intensities_pad_cl.set_scalar_arg_dtypes(
                [None, None, None, None, self.int_t, self.int_t, self.int_t, self.real_t, None, None, None, None,
                 self.int_t])

        if R is None:
            R = np.eye(3)
        R = self.vec16(R.T, dtype=self.real_t)

        nF = self.int_t(nF)
        nS = self.int_t(nS)
        n_pixels = self.int_t(nF * nS)
        if add is True:
            add = 1
        else:
            add = 0
        add = self.int_t(add)

        abc = self.vec16(abc, dtype=self.real_t)
        N = self.vec4(N, dtype=self.int_t)
        T = self.vec4(T, dtype=self.real_t)
        F = self.vec4(F, dtype=self.real_t)
        S = self.vec4(S, dtype=self.real_t)
        B = self.vec4(B, dtype=self.real_t)
        I_dev = self.to_device(I, dtype=self.real_t, shape=(n_pixels))

        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) *
                             self.group_size)
        self.lattice_transform_intensities_pad_cl(self.queue, (global_size,),
                                                  (self.group_size,), abc,
                                                  N, R, I_dev.data, n_pixels,
                                                  nF, nS, w, T, F, S, B, add)
        self.queue.finish()

        if I is None:
            return I_dev.get()
        else:
            return I_dev

    def gaussian_lattice_transform_intensities_pad(self, abc, N, T, F, S, B, nF, nS, w,
                                                   R=None, I=None, add=False):
        r"""
        Calculate crystal lattice transform intensities for a pixel-array detector.  Uses a Gaussian
        approximation to the lattice transform.

        Arguments:
            abc (numpy array) : A 3x3 array containing real-space basis vectors.  Vectors are contiguous
                                in memory.
            N (numpy array)   : An array containing number of unit cells along each of three axes.
            T (numpy array)   : Translation to center of corner pixel.
            F (numpy array)   : Fast-scan basis vector.
            S (numpy array)   : Slow-scan basis vector.
            B (numpy array)   : Incident beam vector.
            nF (int)          : Number of fast-scan pixels.
            nS (int)          : Number of slow-scan pixels.
            w (float)         : Wavelength.
            R (numpy array)   : Rotation matrix acting on atom vectors.
                (we quietly transpose R and let it operate on q-vectors for speedups)
            I (:class:pyopencl.array.Array) : OpenCL device array containing intensities.
            add (bool)        : If true, the function will add to the input I buffer, else the buffer is
                                overwritten.

        Returns:
            If I == None, then the output is a numpy array.  Otherwise, it is an opencl array.
        """

        if not hasattr(self, 'gaussian_lattice_transform_intensities_pad_cl'):
            self.gaussian_lattice_transform_intensities_pad_cl = self.programs.gaussian_lattice_transform_intensities_pad
            self.gaussian_lattice_transform_intensities_pad_cl.set_scalar_arg_dtypes(
                [None, None, None, None, self.int_t, self.int_t, self.int_t, self.real_t, None, None, None, None,
                 self.int_t])

        if R is None:
            R = np.eye(3)
        R = self.vec16(R, dtype=self.real_t)

        nF = self.int_t(nF)
        nS = self.int_t(nS)
        n_pixels = self.int_t(nF * nS)
        if add is True:
            add = 1
        else:
            add = 0
        add = self.int_t(add)

        abc = self.vec16(abc, dtype=self.real_t)
        N = self.vec4(N, dtype=self.int_t)
        T = self.vec4(T, dtype=self.real_t)
        F = self.vec4(F, dtype=self.real_t)
        S = self.vec4(S, dtype=self.real_t)
        B = self.vec4(B, dtype=self.real_t)
        I_dev = self.to_device(I, dtype=self.real_t, shape=(n_pixels))

        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) * self.group_size)
        self.gaussian_lattice_transform_intensities_pad_cl(self.queue, (global_size,),
                                                  (self.group_size,), abc,
                                                  N, R, I_dev.data, n_pixels,
                                                  nF, nS, w, T, F, S, B, add)
        self.queue.finish()

        if I is None:
            return I_dev.get()
        else:
            return I_dev

    def test_atomic_add_real(self, a, b):
        if not hasattr(self, 'test_atomic_add_real_cl'):
            self.test_atomic_add_real_cl = self.programs.test_atomic_add_real
            self.test_atomic_add_real_cl.set_scalar_arg_dtypes([None, None, self.int_t])
        global_size = np.int(np.ceil(len(a) / np.float(self.group_size)) * self.group_size)
        self.test_atomic_add_real_cl(self.queue, (global_size,), (self.group_size,), a.data, b.data, len(a))
        self.queue.finish()

    def test_atomic_add_int(self, a, b):
        if not hasattr(self, 'test_atomic_add_real_cl'):
            self.test_atomic_add_int_cl = self.programs.test_atomic_add_int
            self.test_atomic_add_int_cl.set_scalar_arg_dtypes([None, None, self.int_t])
        n = self.int_t(len(b))
        global_size = np.int(np.ceil(n / np.float(self.group_size)) * self.group_size)
        self.test_atomic_add_int_cl(self.queue, (global_size,), (self.group_size,), a.data, b.data, n)
        self.queue.finish()

    def divide_nonzero_inplace(self, a, b):
        if not hasattr(self, 'divide_nonzero_inplace_real_cl'):
            self.divide_nonzero_inplace_real_cl = self.programs.divide_nonzero_inplace_real
            self.divide_nonzero_inplace_real_cl.set_scalar_arg_dtypes([None, None, self.int_t])
        n = self.int_t(len(a))
        global_size = np.int(np.ceil(n / np.float(self.group_size)) * self.group_size)
        self.divide_nonzero_inplace_real_cl(self.queue, (global_size,), (self.group_size,), a.data, b.data, n)
        self.queue.finish()


class ClCoreDerek(ClCore):

    def __init__(self, *args, **kwargs):

        ClCore.__init__(self, *args, **kwargs)

        # important for comermann pipeline
        self.primed_cromermann = False

        self.qrf_cromer_mann_cl = self.programs.qrf_cromer_mann
        self.qrf_cromer_mann_cl.set_scalar_arg_dtypes([None, None, None, None, None, self.int_t])

    @staticmethod
    def to_device_static(array, dtype, queue):
        """
        Static method

        This is a thin wrapper for pyopencl.array.to_device().  It will convert a numpy
        array into a pyopencl.array and send it to the device memory.  So far this only
        deals with float and comlex arrays, and it should figure out which type it is.

        Arguments:
            array (numpy/cl array; float/complex type): Input array.
            dtype (np.dtype): Specify the desired type in opencl.  The two types that
                               are useful here are np.float32 and np.complex64
            queue, CL queue
        Returns:
            pyopencl array
        """

        # TODO: why does this method exist?  It is not used anywhere.
        if isinstance(array, cl_array):
            return array

        return cl.array.to_device(queue, np.ascontiguousarray(array.astype(dtype)))

    def phase_factor_qrf_inplace(self, q, r, f, R=None, q_is_qdev=False):

        r"""
        Calculate diffraction amplitudes: sum over f_n*exp(-iq.r_n)

        Arguments:
            q (numpy/cl float array [N,3]): Scattering vectors (2\pi/\lambda).
            r (numpy/cl float array [M,3]): Atomic coordinates.
            f (numpy/cl complex array [M]): Complex scattering factors.
            R (numpy array [3,3]): Rotation matrix acting on atom vectors
                (we quietly transpose R and let it operate on q-vectors for speedups)
            a (cl complex array [N]): Optional container for complex scattering
              amplitudes.

        Returns:
            (numpy/cl complex array [N]): Diffraction amplitudes.  Will be a cl array
              if there are input cl arrays.
        """

        if R is None:
            R = np.eye(3, dtype=self.real_t)
        R = self.vec16(R.T, dtype=self.real_t)

        n_pixels = self.int_t(q.shape[0])
        n_atoms = self.int_t(r.shape[0])
        if not q_is_qdev:
            q_dev = self.to_device(q, dtype=self.real_t)
        else:
            q_dev = q
        r_dev = self.to_device(r, dtype=self.real_t)
        f_dev = self.to_device(f, dtype=self.complex_t)
        #R16_dev = self.to_device(R16, dtype=self.real_t)

        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size))
                             * self.group_size)

        add=self.int_t(1) # inplace always adds...
        self.phase_factor_qrf_cl(self.queue, (global_size,),
                                 (self.group_size,), q_dev.data,
                                 r_dev.data,
                                 f_dev.data,
                                 R,
                                 self.a_dev.data,
                                 n_atoms,
                                 n_pixels, add)

    def init_amps(self, Npix):

        r"""

        Initialize amplitudes for cromer-mann simulator as zeros

        Arguments:
            Npix:

        Returns:
            None
        """

        self.a_dev = self.to_device(np.zeros(Npix), dtype=self.complex_t, shape=(Npix))

    def release_amps(self, reset=False):

        r"""

        retrieve scattering amplitudes from cromer-mann simulator

        Arguments:
            reset:
                whether to reset the amplitudes to zeros

        Returns:

        """

        amps = self.a_dev.get()
        if reset:
            self.init_amps(amps.shape[0])
        return amps

    def prime_cromermann_simulator(self, q_vecs, atomic_nums=None, incoherent=False):
        """
        Prepare special array data for cromermann simulation

        Arguments:
            q_vecs (np.ndarray) :
                Npixels x 3 array of cartesian pixels qx, qy, qz
            atomic_num (np.ndarray) :
                Natoms x 1 array of atomic numbers corresponding
                to the atoms in the target
            incoherent bool:
                Whether to make form factors random
        """

        self.q_vecs = q_vecs

        self.Npix = self.int_t(q_vecs.shape[0])

        # allow these to overflow
        self.Nextra_pix = self.int_t(self.group_size - self.Npix % self.group_size)

        if atomic_nums is None:
            if not incoherent:
                self.form_facts_arr = np.ones((self.Npix + self.Nextra_pix, 1), dtype=self.real_t)
            else:
                self.form_facts_arr = 2*np.pi * \
                        np.random.random((self.Npix + self.Nextra_pix, 1)).astype( dtype=self.real_t)
            self.atomIDs = None
            self.Nspecies = 1
            self._load_amp_buffer()
            self.primed_cromermann = True
            return

        croman_coef = refdata.get_cromermann_parameters(atomic_nums)
        form_facts_dict = refdata.get_cmann_form_factors(croman_coef, self.q_vecs)

        lookup = {}  # for matching atomID to atomic number

        self.form_facts_arr = np.zeros(
            (self.Npix + self.Nextra_pix, len(form_facts_dict)), dtype=self.real_t)

        for i, z in enumerate(form_facts_dict):
            lookup[z] = i  # set the key
            self.form_facts_arr[:self.Npix, i] = form_facts_dict[z]

        self.atomIDs = np.array([lookup[z] for z in atomic_nums])

        self.Nspecies = np.unique(atomic_nums).size

        assert (self.Nspecies < 13)  # can easily change this later if necessary...
        # ^ this assertion is so we can pass inputs to GPU as a float16, 3 q vectors and 13 atom species
        # where one is reserved to be a dummie

        #       load the amplitudes
        self._load_amp_buffer()

        self.primed_cromermann = True

    def get_r_cromermann(self, atom_vecs, sub_com=False):

        r"""
        combine atomic vectors and atomic flux factors into an openCL buffer

        Arguments:
            atom_vecs (np.ndarray):
                Atomic positions

            sub_com (bool):
                Whether to sub the center of mass from the atom vecs

        Returns:
            pyopenCL buffer data :
                Natoms x 4 contiguous openCL buffer array
        """

        assert (self.primed_cromermann), "run ClCore.prime_comermann_simulator first"

        if sub_com:
            atom_vecs -= atom_vecs.mean(0)

        self._load_r_buffer(atom_vecs)

        return self.r_buff.data

    def _load_r_buffer(self, atom_vecs):
        """
        makes the r buffer for use in the cromer-mann simulator, where
        r-vector is Nx4, the last dimension being atomic number
        used for lookup of form factor
        """
        if self.atomIDs is not None:
            self.r_vecs = np.concatenate(
                (atom_vecs, self.atomIDs[:, None]), axis=1)
        else:
            self.r_vecs = np.concatenate(
                (atom_vecs, np.zeros((atom_vecs.shape[0], 1))), axis=1)

        self.Nato = self.r_vecs.shape[0]

        self.r_buff = self.to_device(self.r_vecs, dtype=self.real_t)

    def get_q_cromermann(self):
        """
        combine form factors and q-vectors and load onto a CL buffer

        Arguments:

            q_vecs (np.ndarray) :
                Npixels x 3 array (inverse angstroms)

            atomic_nums (np.ndarray) :
                Natoms x 1 array of atomic numbers

        Returns:
            pyopenCL buffer data :
                Npixelbuff x 16 contiguous openCL buffer array
                where Npixel buff is the first multiple of
                group_size that is greater than Npixels

        """

        assert (self.primed_cromermann), "run ClCore.prime_comermann_simulator first"

        #       load onto device
        self._load_q_buffer()

        return self.q_buff.data

    def _load_q_buffer(self):
        """
        makes the q_buffer so that is it integer mutiple of group size
        this is for the cromer-mann simnulator
        """
        q_zeros = np.zeros((self.Npix + self.Nextra_pix, 16))
        q_zeros[:self.Npix, :3] = self.q_vecs
        q_zeros[:, 3:3 + self.Nspecies] = self.form_facts_arr
        self.q_buff = self.to_device(q_zeros, dtype=self.real_t)

    def _load_amp_buffer(self):
        """
        makes the amplitude buffer so that it is integer multiple of groupsize
        """
        #make output buffer; initialize as 0s
        self.A_buff = self.to_device(
            np.zeros(self.Npix + self.Nextra_pix), dtype=self.complex_t)

        self._A_buff_data = self.A_buff.data

    def run_cromermann(self, q_buff_data, r_buff_data,
                    rand_rot=False, force_rot_mat=None, com=None):

        r"""
        Run the cromer-mann form-factor simulator.

        Arguments
            q_buff_data (pyopenCL buffer data) :
                should have shape NpixelsCLx16 where
                NpixelsCL is the first multiple of group_size greater than
                Npixels.
                Use :func:`get_group_size` to check the currently
                set group_size. The data stored in
                q[Npixels,:3] should be the q-vectors.
                The data stored in q[Npixels,3:Nspecies] should be the
                q-dependent atomic form factors for up to Nspecies=13
                atom species See :func:`prime_comermann_simulator`
                for details regarding the form factor storage and atom
                species identifier

            r_buff_data (pyopenCL buffer data) :
                Should have shape Natomsx4. The data stored in
                r_buff_data[:,:3] are the atomic positions in cartesian
                (x,y,z).  The data stored in r_buff_data[:,3] are
                the atom species identifiers (0,1,..Nspecies-1)
                mapping the atom species here to the form factor value
                in q_buff_data.

            rand_rot (bool) :
                Randomly rotate the molecule

            force_rand_rot (np.ndarray) :
                Supply a specific rotation matrix that operates on molecules

            com (np.ndarray) :
                Offset the center of mass of the molecule

        .. note::
            For atom r_i the atom species identifier is sp_i =
            r_buff_data[r_i,3].
            Then, for pixel q_i, the simulator can find the corresponding
            form factor in q_buff_dat[q_i,3+sp_i].
            I know it is confusing, but it's efficient.
        """

        #       set the rotation
        if rand_rot:
            self.rot_mat = ba.utils.random_rotation().ravel().astype(self.real_t)
        elif force_rot_mat is not None:
            self.rot_mat = force_rot_mat.astype(self.real_t)
        else:
            self.rot_mat = np.eye(3).ravel().astype(self.real_t)

        self._set_rand_rot()

        #       set the center of mass
        if com is not None:
            self.com_vec = com.astype(self.real_t)
        else:
            self.com_vec = np.zeros(3).astype(self.real_t)
        self._set_com_vec()

        #       run the program
        self.qrf_cromer_mann_cl( self.queue, (int(self.Npix + self.Nextra_pix),),
            (self.group_size,), q_buff_data, r_buff_data,
            self.rot_buff.data, self.com_buff.data,
            self._A_buff_data, self.Nato)

    def _set_rand_rot(self):
        r"""Sets the random rotation matrix on device"""

        self.rot_buff = self.to_device(self.rot_mat, dtype=self.real_t)

    def _set_com_vec(self):
        """sets the center-of mass vectors on the device"""
        self.com_buff = self.to_device(self.com_vec, dtype=self.real_t)

    def release_amplitudes(self, reset=False):
        r"""
        Releases the amplitude buffer from the GPU

        Arguments:
            reset (bool) : Reset the amplitude buffer to 0's on the GPU

        Returns (np.ndarray) : Scattering amplitudes
        """

        Amps = self.A_buff.get()[:-self.Nextra_pix]
        if reset:
            self._load_amp_buffer()
        return Amps



def helpme():

    r"""

    Same as helpme() function.

    """

    help()

def help():

    r"""
    Print out some useful information about platforms and devices that are
    available for running simulations.
    """

    def print_info(obj, info_cls):
        for info_name in sorted(dir(info_cls)):
            if not info_name.startswith("_") and info_name != "to_string":
                info = getattr(info_cls, info_name)
                try:
                    info_value = obj.get_info(info)
                except:
                    info_value = "<error>"

                if (info_cls == cl.device_info and info_name == "PARTITION_TYPES_EXT"
                    and isinstance(info_value, list)):
                    print("%s: %s" % (info_name, [
                        cl.device_partition_property_ext.to_string(v,
                                                    "<unknown device partition property %d>")
                        for v in info_value]))
                else:
                    try:
                        print("%s: %s" % (info_name, info_value))
                    except:
                        print("%s: <error>" % info_name)

    short = False

    for platform in cl.get_platforms():
        print(75 * "=")
        print(platform)
        print(75 * "=")
        if not short:
            print_info(platform, cl.platform_info)

        for device in platform.get_devices():
            if not short:
                print(75 * "-")
            print(device)
            if not short:
                print(75 * "-")
                print_info(device, cl.device_info)
                ctx = cl.Context([device])
                for mf in [
                    cl.mem_flags.READ_ONLY,
                    # cl.mem_flags.READ_WRITE,
                    # cl.mem_flags.WRITE_ONLY
                ]:
                    for itype in [
                        cl.mem_object_type.IMAGE2D,
                        cl.mem_object_type.IMAGE3D
                    ]:
                        try:
                            formats = cl.get_supported_image_formats(ctx, mf, itype)
                        except:
                            formats = "<error>"
                        else:
                            def str_chd_type(chdtype):
                                result = cl.channel_type.to_string(chdtype,
                                                                   "<unknown channel data type %d>")

                                result = result.replace("_INT", "")
                                result = result.replace("UNSIGNED", "U")
                                result = result.replace("SIGNED", "S")
                                result = result.replace("NORM", "N")
                                result = result.replace("FLOAT", "F")
                                return result

                            formats = ", ".join(
                                "%s-%s" % (
                                    cl.channel_order.to_string(iform.channel_order,
                                                               "<unknown channel order 0x%x>"),
                                    str_chd_type(iform.channel_data_type))
                                for iform in formats)

                        print("%s %s FORMATS: %s\n" % (
                            cl.mem_object_type.to_string(itype),
                            cl.mem_flags.to_string(mf),
                            formats))
                del ctx

    print('')
    print('')
    print('')
    print('')
    print('Summary of platforms and devices (see details above):')
    print(75 * "=")
    i = 0
    for platform in cl.get_platforms():
        print(i, platform)
        i += 1
        j = 0
        for device in platform.get_devices():
            print(2 * '-', j, device)
            j += 1
    print(75 * "=")
    print('')
    print("You can set the environment variable PYOPENCL_CTX to choose the ")
    print("device and platform automatically.  For example,")
    print("> export PYOPENCL_CTX='1'")


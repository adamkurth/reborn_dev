"""
This module contains some core functions that are useful for simulating
diffraction on GPU devices.  It is not finished yet...

Some environment variables that affect this module:
'BORNAGAIN_CL_GROUPSIZE' : This sets the default groupsize.  It will
otherwise be 32.  If you are using a CPU you might want to set 
BORNAGAIN_CL_GROUPSIZE=1 .
"""

import time
import sys
import os
import pkg_resources

import numpy as np
import refdata
import bornagain as ba
import pyopencl as cl
import pyopencl.array
#from __builtin__ import None    # this is causing problems for some reason...

clcore_file = pkg_resources.resource_filename(
    'bornagain.simulate', 'clcore.cpp')


class ClCore(object):
    
    def __init__(self,context=None,queue=None,group_size=None,
                 double_precision=False):
        
        self.group_size = None
        self.programs = None
        self.double_precision = double_precision

        
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
       
        # Abstract real and complex types to allow for double/single
        self._setup_precision(double_precision)

        # Setup the group size.
        self.set_groupsize(group_size)
        
        # setup the programs
        self._load_programs()

        # important for comermann pipeline
        self.primed_cromermann=False

    def set_groupsize(self, group_size):
        """
        If the environment variable BORNAGAIN_CL_GROUPSIZE is set then use
        that value.
        
        If the group size exceeds the max allowed group size, then make it
        smaller (but print warning)
        """
        if os.environ.get('BORNAGAIN_CL_GROUPSIZE') is not None:
            group_size = np.int(os.environ.get('BORNAGAIN_CL_GROUPSIZE'))
        if group_size is None:
            group_size = 32
        max_group_size = self.queue.device.max_work_group_size
        if self.double_precision:
            max_group_size = int(max_group_size/2)
        if group_size > max_group_size:
            sys.stderr.write('Changing group size from %d to %d.\n'
                     'Set BORNAGAIN_CL_GROUPSIZE=%d to avoid this error.\n' 
                     % (group_size, max_group_size, max_group_size))
            group_size = max_group_size
        self.group_size = group_size
        
    def _double_precision_is_available(self):
#         print(self.queue.platform)
        if 'cl_khr_fp64' not in self.queue.device.extensions.split():
            return False
# TODO: fix stupid errors to do with Apple's CL double implementation?  Why
#       doesn't double work on apple?
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
        # TODO: Decide if integers should be made double also.  As of now, they
        #       are all single precision.
        self.int_t = np.int32
        self.real_t = np.float64
        self.complex_t = np.complex128
        
    def _use_float(self):
        self.int_t = np.int32
        self.real_t = np.float32
        self.complex_t = np.complex64
    
    def _load_programs(self):
        self._build_openCL_programs()
        self._load_get_group_size()
        self._load_phase_factor_qrf()
        self._load_phase_factor_pad()
        self._load_phase_factor_mesh()
        self._load_buffer_mesh_lookup()
        self._load_mod_squared_complex_to_real()
        self._load_qrf_default()
        self._load_qrf_kam()
        self._load_lattice_transform_intensities_pad()

    def _build_openCL_programs(self):
        clcore_file = pkg_resources.resource_filename(
            'bornagain.simulate', 'clcore.cpp')
        kern_str = open(clcore_file).read()
        if self.double_precision:
            kern_str = kern_str.replace("float", "double")
        build_opts = ['-D', 'GROUP_SIZE=%d' % self.group_size]
        self.programs = cl.Program(self.context, kern_str).build(options=build_opts)
    
    def _load_get_group_size(self):
        # Configure the python interface to the cl programs
        self.get_group_size_cl = self.programs.get_group_size
        self.get_group_size_cl.set_scalar_arg_dtypes([None])
    
    def _load_phase_factor_qrf(self):
        self.phase_factor_qrf_cl = self.programs.phase_factor_qrf
        self.phase_factor_qrf_cl.set_scalar_arg_dtypes(
                    [None, None, None, None, None, self.int_t, self.int_t])
    
    def _load_phase_factor_pad(self):
        self.phase_factor_pad_cl = self.programs.phase_factor_pad
        self.phase_factor_pad_cl.set_scalar_arg_dtypes(
             [None, None, None, None, self.int_t, self.int_t, self.int_t, self.int_t,
              self.real_t, None, None, None, None])
    
    def _load_phase_factor_mesh(self):
        self.phase_factor_mesh_cl = self.programs.phase_factor_mesh
        self.phase_factor_mesh_cl.set_scalar_arg_dtypes(
             [None, None, None, self.int_t, self.int_t, None, None, None])

    def _load_buffer_mesh_lookup(self):
        self.buffer_mesh_lookup_cl = self.programs.buffer_mesh_lookup
        self.buffer_mesh_lookup_cl.set_scalar_arg_dtypes(
             [None, None, None, self.int_t, None, None, None, None])

    def _load_lattice_transform_intensities_pad(self):
        self.lattice_transform_intensities_pad_cl = \
              self.programs.lattice_transform_intensities_pad
        self.lattice_transform_intensities_pad_cl.set_scalar_arg_dtypes(
             [None, None, None, None, self.int_t, self.int_t, self.int_t, 
              self.real_t, None, None, None, None, self.int_t])

    def _load_mod_squared_complex_to_real(self):
        self.mod_squared_complex_to_real_cl = self.programs.mod_squared_complex_to_real
        self.mod_squared_complex_to_real_cl.set_scalar_arg_dtypes(
             [None, None, self.int_t])
    
    def _load_qrf_default(self):
        self.qrf_default_cl = self.programs.qrf_default
        self.qrf_default_cl.set_scalar_arg_dtypes(
            [None, None, None, None,  self.int_t])
    
    def _load_qrf_kam(self):
        self.qrf_kam_cl = self.programs.qrf_kam
        self.qrf_kam_cl.set_scalar_arg_dtypes(
            [None, None, None, None,None, self.int_t])
    
    def vec4(self, x, dtype=None):
        """
        Evdidently pyopencl does not deal with 3-vectors very well, so we use
        4-vectors and pad with a zero at the end.
            
        Arguments:
            - x, np.ndarray 
            
            - dtype, np.dtype 
                default is np.float32
            
        Returns:
            - numpy array of length 4
        """
        
        if dtype is None:
            dtype = self.real_t
        return np.array([x.flat[0], x.flat[1], x.flat[2], 0.0], dtype=dtype)
 

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

        if isinstance(array, cl.array.Array):
            return array

        return cl.array.to_device(queue, 
                                  np.ascontiguousarray(array.astype(dtype)))


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
        """
        retrieve the currently set group_size
        """
        group_size_dev = self.to_device(np.zeros((1)), dtype=self.int_t)
        self.get_group_size_cl(self.queue, (self.group_size,), 
                               (self.group_size,), group_size_dev.data)
        
        return group_size_dev.get()[0]


    def mod_squared_complex_to_real(self,A,I):
        '''
        Compute the real-valued modulus square of complex numbers.
        Good example of a function that shouldn't exist, but I needed to add 
        it here because the pyopencl.array.Array class fails at this seemingly
        simple task.
        '''
        
        A_dev = self.to_device(A, dtype=self.complex_t)
        I_dev = self.to_device(I, dtype=self.real_t)
        n = self.int_t(np.prod(A.shape))
        
        global_size = np.int(np.ceil(n / np.float(self.group_size)) 
                             * self.group_size)
        
        self.mod_squared_complex_to_real_cl(self.queue, (global_size,), 
                                 (self.group_size,), A.data, I.data, n)


    def phase_factor_qrf_inplace(self, q, r, f, R=None):
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
        R16 = np.zeros(16, dtype=self.real_t)
        R16[0:9] = R.ravel()
    
        n_pixels = self.int_t(q.shape[0])
        n_atoms = self.int_t(r.shape[0])
        q_dev = self.to_device(q, dtype=self.real_t)
        r_dev = self.to_device(r, dtype=self.real_t)
        f_dev = self.to_device(f, dtype=self.complex_t)
        R16_dev = self.to_device(R16, dtype=self.real_t)
    
        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) 
                             * self.group_size)
    
        self.phase_factor_qrf_cl(self.queue, (global_size,), 
                                 (self.group_size,), q_dev.data, r_dev.data, 
                                 f_dev.data, R16_dev.data, self.a_dev.data, n_atoms,
                                 n_pixels)
    
    def next_multiple_groupsize(self, N):
        if N % self.group_size >0:
            return self.int_t(self.group_size - N % self.group_size)
        else:
            return 0

    def init_amps(self, Npix):
        self.a_dev = self.to_device( np.zeros( Npix), dtype=self.complex_t, shape=(Npix))
        
    def release_amps(self, reset=False):
        amps = self.a_dev.get()
        if reset:
            self.init_amps(amps.shape[0])
        return amps

    
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
    
        for i in range(0,10):
            global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) 
                                 * self.group_size)
        
            R16_dev = self.to_device(R16, dtype=self.real_t)
            self.phase_factor_qrf_cl(self.queue, (global_size,), 
                                     (self.group_size,), q_dev.data, r_dev.data, 
                                     f_dev.data, R16_dev.data, a_dev.data, n_atoms,
                                     n_pixels)
    
        if a is None:
            return a_dev.get()
        else:
            return a_dev
        
    def phase_factor_pad(self, r, f, T, F, S, B, nF, nS, w, R=None, a=None):
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
    
        nF = self.int_t(nF)
        nS = self.int_t(nS)
        n_pixels = self.int_t(nF * nS)
        n_atoms = self.int_t(r.shape[0])
        r_dev = self.to_device(r, dtype=self.real_t)
        f_dev = self.to_device(f, dtype=self.complex_t)
        R_dev = self.to_device(R, dtype=self.real_t)
        
        T_dev = self.to_device( T, dtype=self.real_t)
        F_dev = self.to_device( F, dtype=self.real_t)
        S_dev = self.to_device( S, dtype=self.real_t)
        B_dev = self.to_device( B, dtype=self.real_t)
        
        a_dev = self.to_device(a, dtype=self.complex_t, shape=(n_pixels))
    
        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) * 
                             self.group_size)
    
        
        print('global,group',global_size, self.group_size)

        self.phase_factor_pad_cl(self.queue, (global_size,), 
                                 (self.group_size,), r_dev.data,
                            f_dev.data, R_dev.data, a_dev.data, n_pixels, n_atoms,
                            nF, nS, w, T_dev.data, F_dev.data, S_dev.data, B_dev.data)
    
        if a is None:
            return a_dev.get()
        else:
            return a_dev

    def phase_factor_mesh(self, r, f, N, q_min, q_max, a=None):
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

        Returns:
            An array of complex scattering amplitudes.  By default this is a normal
               numpy array.  Optionally, this may be an opencl buffer.
        '''
    
        N = np.array(N, dtype=self.int_t)
        q_max = np.array(q_max, dtype=self.real_t)
        q_min = np.array(q_min, dtype=self.real_t)
    
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
                                  q_min)
    
        if a is None:
            return a_dev.get()
        else:
            return a_dev

    def buffer_mesh_lookup(self, a_map, N, q_min, q_max, q, R=None, 
                           a_out=None):
        """
        This is supposed to lookup intensities from a 3d mesh of amplitudes.

        Arguments:
            a_map (numpy array): Complex scattering amplitudes (usually generated from 
               the function phase_factor_mesh())
            N (int): As defined in phase_factor_mesh()
            q_min (float): As defined in phase_factor_mesh()
            q_max (float): As defined in phase_factor_mesh()
            q (Nx3 numpy array): q-space coordinates at which we want to interpolate
               the complex amplitudes in a_dev
            R (3x3 numpy array): Rotation matrix that will act on the q vectors
            a_out: (clarray) The output array (optional)

        Returns:
            numpy array of complex amplitudes
        """
    
        if R is None:
            R = np.eye(3, dtype=self.real_t).ravel()
        
        R16 = np.zeros([16], dtype=self.real_t)
        R16[0:9] = R.flatten().astype(self.real_t)
    
        N = np.array(N, dtype=self.int_t)
        q_max = np.array(q_max, dtype=self.real_t)
        q_min = np.array(q_min, dtype=self.real_t)
    
        if len(N.shape) == 0:
            N = (np.ones(3)*N).astype(self.int_t)
        if len(q_max.shape) == 0:
            q_max = self.real_t(np.ones(3)*q_max)
        if len(q_min.shape) == 0:
            q_min = self.real_t(np.ones(3)*q_min)
    
        deltaQ = np.array((q_max - q_min) / (N - 1.0), dtype=self.real_t)
    
        n_pixels = self.int_t(q.shape[0])
    
        a_map_dev = self.to_device(a_map, dtype=self.complex_t)
        q_dev = self.to_device(q, dtype=self.real_t)
        N = self.vec4(N, dtype=self.int_t)
        deltaQ = self.vec4(deltaQ, dtype=self.real_t)
        q_min = self.vec4(q_min, dtype=self.real_t)
        a_out_dev = self.to_device(
            a_out, dtype=self.complex_t, shape=(n_pixels))
    
        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) 
                             * self.group_size)
   
        R16_dev = self.to_device(R16, dtype=self.real_t)
        self.buffer_mesh_lookup_cl(self.queue, (global_size,), (self.group_size,), 
                              a_map_dev.data, q_dev.data, a_out_dev.data, 
                              n_pixels, N, deltaQ, q_min, R16_dev.data)
    
        if a_out is None:
            return a_out_dev.get()
        else:
            return a_out_dev


    def lattice_transform_intensities_pad(self, abc, N, T, F, S, B, nF, nS, w, 
                                          R=None, I=None, add=False):
        """
        This is not documentation.  That is Rick's fault.
        """
    
        if R is None:
            R = np.eye(3, dtype=self.real_t)
    
        nF = self.int_t(nF)
        nS = self.int_t(nS)
        n_pixels = self.int_t(nF * nS)
        if add is True:
            add = 1
        else:
            add = 0
        add = self.int_t(add)
        
        abc_dev = self.to_device(abc, dtype=self.real_t)
        N_dev = self.to_device(N, dtype=self.int_t)
        R_dev = self.to_device(R, dtype=self.real_t)
        T_dev = self.to_device(T, dtype=self.real_t)
        F_dev = self.to_device(F, dtype=self.real_t)
        S_dev = self.to_device(S, dtype=self.real_t)
        B_dev = self.to_device(B, dtype=self.real_t)
        I_dev = self.to_device(I, dtype=self.real_t, shape=(n_pixels))
    
        global_size = np.int(np.ceil(n_pixels / np.float(self.group_size)) * 
                             self.group_size)
        self.lattice_transform_intensities_pad_cl(self.queue, (global_size,), 
                                 (self.group_size,), abc_dev.data,
                            N_dev.data, R_dev.data, I_dev.data, n_pixels, 
                            nF, nS, w, T_dev.data, F_dev.data, S_dev.data, B_dev.data, add)
    
        if I is None:
            return I_dev.get()
        else:
            return I_dev


    def prime_cromermann_simulator(self, q_vecs, atomic_nums=None):
        """
        Prepare special array data for cromermann simulation

        Arguments
            - q_vecs, np.ndarray
                Npixels x 3 array of cartesian pixels qx, qy, qz
            - atomic_num, np.ndarray
                Natoms x 1 array of atomic numbers corresponding
                to the atoms in the target
        """

        self.q_vecs = q_vecs

        self.Npix = self.int_t(q_vecs.shape[0])

#       allow these to overflow
        self.Nextra_pix = self.int_t(self.group_size - self.Npix % self.group_size)

        if atomic_nums is None:
            self.form_facts_arr = np.ones((self.Npix+self.Nextra_pix,1), dtype=self.real_t)
            self.atomIDs = None
            self.Nspecies = 1
            self._load_amp_buffer()
            self.primed_cromermann = True
            return

        croman_coef = refdata.get_cromermann_parameters(atomic_nums)
        form_facts_dict = refdata.get_cmann_form_factors(croman_coef, self.q_vecs)

        lookup = {}  # for matching atomID to atomic number
        
        self.form_facts_arr = np.zeros(
            (self.Npix+self.Nextra_pix, len(form_facts_dict)), dtype=self.real_t)
        
        for i, z in enumerate(form_facts_dict):
            lookup[z] = i  # set the key
            self.form_facts_arr[:self.Npix,i] = form_facts_dict[z]
        
        self.atomIDs = np.array([lookup[z] for z in atomic_nums])


        self.Nspecies = np.unique( atomic_nums).size

        assert( self.Nspecies < 13) # can easily change this later if necessary... 
        #^ this assertion is so we can pass inputs to GPU as a float16, 3 q vectors and 13 atom species
        # where one is reserved to be a dummie 

#       load the amplitudes
        self._load_amp_buffer()

        self.primed_cromermann = True

    def get_r_cromermann(self, atom_vecs, sub_com=False):
        """
        combine atomic vectors and atomic flux factors
        into an openCL buffer

        Arguments,
            - atom_vecs, np.ndarray
                atomic positions
            - sub_com, bool,
                whether to sub the center of mass from the atom vecs
        
        Returns
            - pyopenCL buffer data
                Natoms x 4 contiguous openCL buffer array
        """
       
        assert(self.primed_cromermann), "run ClCore.prime_comermann_simulator first"

        if sub_com:
            atom_vecs -= atom_vecs.mean(0)
        
        self._load_r_buffer(atom_vecs)

        return self.r_buff.data
        
    def _load_r_buffer(self, atom_vecs):
    
        if self.atomIDs is not None:
            self.r_vecs = np.concatenate(
                (atom_vecs, self.atomIDs[:, None]), axis=1)
        else:
            self.r_vecs = np.concatenate(
                (atom_vecs, np.zeros( (atom_vecs.shape[0],1))), axis=1)

        self.Nato = self.r_vecs.shape[0]

        self.r_buff = self.to_device(self.r_vecs, dtype=self.real_t)


    def get_q_cromermann(self):
        """ 
        combine form factors and q-vectors and load onto a CL buffer
        
        Arguments
            - q_vecs, np.ndarray
                Npixels x 3 array (inverse angstroms)

            - atomic_nums, np.ndarray
                Natoms x 1 array of atomic numbers
        
        Returns
            - pyopenCL buffer data
                Npixelbuff x 16 contiguous openCL buffer array
                where Npixel buff is the first multiple of 
                group_size that is greater than Npixels
        """
        
        assert(self.primed_cromermann), "run ClCore.prime_comermann_simulator first"

#       load onto device
        self._load_q_buffer()
    
        return self.q_buff.data

    def _load_q_buffer(self):
        q_zeros = np.zeros((self.Npix+self.Nextra_pix, 16))
        q_zeros[:self.Npix, :3] = self.q_vecs
        q_zeros[:,3:3+self.Nspecies] = self.form_facts_arr
        self.q_buff = self.to_device( q_zeros, dtype=self.real_t)

    def _load_amp_buffer(self):
#       make output buffer; initialize as 0s
        self.A_buff = self.to_device(
            np.zeros(self.Npix+self.Nextra_pix), dtype=self.complex_t)

        self._A_buff_data = self.A_buff.data

    def run_cromermann(self, q_buff_data, r_buff_data, rand_rot=False, force_rot_mat=None, com=None):
        """
        Run the qrf kam simulator

        Arguments
            - q_buff_data, pyopenCL buffer data
                should have shape NpixelsCLx16 where NpixelsCL
                is the first multiple of group_size greater than 
                Npixels. Use :func:`get_group_size` to check the currently
                set group_size
                - The data stored in q[Npixels,:3] should be the q-vectors
                - The data stored in q[Npixels,3:Nspecies] should be the q-dependent 
                    atomic form factors for up to Nspecies=13 atom species 
                    See :func:`prime_comermann_simulator` for details regarding the 
                    form factor storage and atom species identifier
            
            - r_buff_data, pyopenCL buffer data
                should have shape Natomsx4
                - The data stored in r_buff_data[:,:3] are the atomic positions in cartesian (x,y,z)
                - The data stored in r_buff_data[:,3] are the atom species identifiers (0,1,..Nspecies-1)
                    mapping the atom species here to the form factor value in q_buff_data.
            
            - rand_rot, bool
                whether to randomly rotate the molecule
            
            - force_rand_rot, np.ndarray
                supply a specific rotation matrix
            
            - com, np.ndarray
                offset the center of mass of the molecule
                
        .. note::
            For atom r_i the atom species identifier is sp_i = r_buff_data[r_i,3].
            Then, for pixel q_i, the simulator can find the corresponding form factor in
            q_buff_dat[q_i,3+sp_i]. I know it is confusing, but it's efficient.
        """

#       set the rotation
        if rand_rot:
            self.rot_mat = ba.utils.random_rotation_matrix().ravel().astype(self.real_t)
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
        self.qrf_kam_cl( self.queue, (int(self.Npix+self.Nextra_pix),),(self.group_size,),
                         q_buff_data, r_buff_data, self.rot_buff.data, 
                         self.com_buff.data, self._A_buff_data, self.Nato)

    def _set_rand_rot(self):
        self.rot_buff = self.to_device(self.rot_mat, dtype=self.real_t)
    
    def _set_com_vec(self):
        self.com_buff = self.to_device(self.com_vec, dtype=self.real_t)

    def release_amplitudes(self, reset=False):
        """
        Releases the amplitude buffer from the GPU
        
        Arguments
            - reset, bool
                reset the amplitude buffer to 0's on the GPU

        Returns
            - np.ndarray
                scattering amplitudes
        """
        Amps = self.A_buff.get() [:-self.Nextra_pix]
        if reset:
            self._load_amp_buffer()
        return Amps


def helpme():

    """
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
    
    short=False
    
    for platform in cl.get_platforms():
        print(75*"=")
        print(platform)
        print(75*"=")
        if not short:
            print_info(platform, cl.platform_info)
    
        for device in platform.get_devices():
            if not short:
                print(75*"-")
            print(device)
            if not short:
                print(75*"-")
                print_info(device, cl.device_info)
                ctx = cl.Context([device])
                for mf in [
                        cl.mem_flags.READ_ONLY,
                        #cl.mem_flags.READ_WRITE,
                        #cl.mem_flags.WRITE_ONLY
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
    print(75*"=")
    i=0
    for platform in cl.get_platforms():
        print(i,platform)
        i+=1
        j=0
        for device in platform.get_devices():
            print(2*'-',j,device)
            j += 1
    print(75*"=")
    print('')
    print("You can set the environment variable PYOPENCL_CTX to choose the ")
    print("device and platform automatically.  For example,")
    print("> export PYOPENCL_CTX='1'")
    

def test():

    import pkg_resources
    import bornagain.target.crystal as crystal
    from bornagain import Molecule
    pdb = pkg_resources.resource_filename('bornagain', '').replace('bornagain/bornagain','bornagain/examples/data/pdb/2LYZ.pdb')
    mol = Molecule(pdb) 
    form_facts = np.ones( mol.atom_vecs.shape[0], np.complex64)
    
    import time
    n_pixels = 2048
    D = ba.detector.SimpleDetector(n_pixels=n_pixels) 
    print ("\tSimulating into %d pixels"%D.Q.shape[0])
    
#   test q-independent
    core = ClCore(double_precision=True)
    Npix = D.n_pixels 
    q = core.to_device(D.Q)
    r = core.to_device( mol.atom_vecs, dtype=core.real_t)
    ff = np.zeros( mol.atom_vecs.shape[0], dtype=core.complex_t)
    ff.real = 1
    f = core.to_device( ff , dtype=core.complex_t)
    
    core.init_amps(Npix)
    print("Testing phase_factor_qrf")
    t = time.time()
    core.phase_factor_qrf_inplace(q,r,f)
    A = core.release_amps(reset=True)
    print ("\tTook %f.4 seconds"%(time.time() - t))
    _ = D.readout(A)
    D.display()

#   now test the cromermann simulation
    print("Testing cromermann")
    core.prime_cromermann_simulator(D.Q, None)
    q = core.get_q_cromermann()
    
    t = time.time()
    r = core.get_r_cromermann(mol.atom_vecs, sub_com=False) 
    core.run_cromermann(q, r, rand_rot=False)
    A2 = core.release_amplitudes()
    print ("\tTook %f.4 seconds"%(time.time() - t))
    _ = D.readout(A2)
    D.display()
  
#   there is slightttt difference between the two methods at low q, not sure why... 
    _ = D.readout( A-A2)
    D.display()

    print("Passed testing mode!")

if __name__=="__main__":
    test()




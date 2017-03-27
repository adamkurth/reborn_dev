import time
import pkg_resources

import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array


import bornagain as ba
import bornagain.simulate.clcore as clcore
import bornagain.simulate.refdata as refdata
import bornagain.target.crystal as crystal


pl = ba.detector.PanelList()
nPixels = 2000
pixelSize = 50e-6
detectorDistance = 0.05
wavelength = 1 #1e-10

# make a single panel detector:
pl.simple_setup(nPixels, nPixels+1, pixelSize, detectorDistance, wavelength)
q_vecs = pl[0].Q # q vectors!

# shape of the 2D det panel (2D image)
img_sh = (pl[0].nS, pl[0].nF)

# Load atom information
#~~~~~~~~~~~~~~~~~~~~~~
pdbFile = '/home/dermen/.local/ext/bornagain/examples/data/pdb/2LYZ.pdb'  
cryst = crystal.structure(pdbFile)
atom_vecs = cryst.r*1e10 # atom positions!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##################
# make a lattice #
##################

#  unit cell edges
a = np.array([79.1, 0, 0]) # angstroms
b = np.array([0, 79.1, 0])
c = np.array([0, 0, 37.9])

n_unit = 30 # set number of unit cells
# lattice coordinates
lat_vecs = np.array( [ i*a + j*b + k*c 
    for i in xrange( n_unit) 
        for j in xrange( n_unit) 
            for k in xrange(n_unit) ] )

# sphericalize the lattice.. 
lat_vecs = ba.utils.sphericalize( lat_vecs)


class ThornAgain:
    def __init__(self, q_vecs, atom_vecs, atomic_nums=None, load_default=False):
        self.q_vecs = q_vecs.astype(float32)
        self.atom_vecs = atom_vecs.astype(float32)
        self.atomic_nums = atomic_nums

#       set dimensions
        self.Nato = np.int32(atom_vecs.shape[0])
        self.Npix = np.int32(q_vecs.shape[0])
        
        self._make_croman_data()
        self._setup_openCL()
        self._load_sources()

        if load_default:
            self.load_program()
        
    def _make_croman_data(self):
        if self.atomic_nums is None:
            self.form_facts_arr = np.ones( (1,self.Npix), dtype=np.float32)
            self.atomIDs = np.zeros( self.Nato, dtype=np.int32)
            return
        
        croman_coef = refdata.get_cromermann_parameters(self.atomic_nums) # 
        form_facts_dict = refdata.get_cmann_form_factors(croman_coef, self.q_vecs) # form factors!

        lookup = {} # for matching atomID to atomic number
        self.form_facts_arr = np.zeros( (len(form_facts_dict), self.q_vecs.shape[0] ), dtype=np.float32  )
        for i,z in enumerate(form_facts_dict):
            lookup[z] = i # set the key
            self.form_facts_arr[i] = form_facts_dict[z]
        self.atomIDs = np.array( [ lookup[z] for z in self.atomic_nums] ).astype(np.int32) # index 0,1, ... , N_unique_atoms

    def _setup_openCL(self):
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        
        self.group_size = clcore.group_size
        self.context = cl.Context(devices=my_gpu_devices)
        self.queue = cl.CommandQueue(self.context)

    def load_program(self, which='default'):
        if which=='default':
            self.prg = self.all_prg.qrf_default
            self.prg.set_scalar_arg_dtypes([None,None,None,None,None, None, np.int32,np.int32])
            self._prime_buffers_default()

    def _prime_buffers_default(self):
#       allow these to overflow (not sure if necess.)
        self.Nextra_pix = (self.group_size - self.Npix%self.group_size)

        mf = cl.mem_flags # how to handle the buffers

#       make a rotation mat
        self.rot_mat = np.eye(3).ravel().astype(np.float32)

#       create openCL data buffers
#       inputs
        self.q_buff = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.q_vecs)
        self.r_buff = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.atom_vecs)
        self.form_facts_buff = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
            hostbuf=self.form_facts_arr)
        self.atomID_buff = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.atomIDs)
        self.rot_buff = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.rot_mat)
#       outputs 
        self.A_buff = clcore.to_device( self.queue, None, shape=(self.Npix+self.Nextra_pix), dtype=np.complex64 )
    
        self.prg_args = [ self.queue, (self.Npix,), None, self.q_buff, self.r_buff, self.form_facts_buff, 
            self.atomID_buff, self.rot_buff, self.A_buff.data, self.Npix, self.Nato ]


    def _set_rand_rot(self, rand_rot):
        if rand_rot:
            self.rot_mat = ba.utils.random_rotation_matrix().T.astype(np.float32)
        else:
            self.rot_mat = np.eye(3).ravel().astype(np.float32)
        
        self.rot_buff = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.rot_mat)
        self.prg_args[7] = self.rot_buff # consider kwargs ?

    def run(self, rand_rot=False):
#       run the program
        self._set_rand_rot(rand_rot)

        
        t = time.time()
        self.prg( *self.prg_args) 
        print ("Took %.4f sec to run the kernel"%float(time.time()-t))

#       copy from device to host
        t = time.time()
        A = self.A_buff.get()[:-self.Nextra_pix]
        
        #a = self.Areal_buff.get()[:-self.Nextra_pix]
        #b = self.Aimag_buff.get()[:-self.Nextra_pix]
        #cl.enqueue_copy(self.queue, self.Areal, self.Areal_buff)
        #cl.enqueue_copy(self.queue, self.Aimag, self.Aimag_buff)
        print ("Took %.4f sec to copy data from device to host"%float(time.time()-t))

#       structure factor
        return np.abs(A)**2 #(self.Areal[:-self.Nextra_pix])**2 + (self.Aimag[:-self.Nextra_pix])**2 


    def _load_sources(self):
        clcore_file = pkg_resources.resource_filename('bornagain.simulate','clcore.cpp')
        with open(clcore_file,'r') as f:
            kern_src = f.read()
        self.all_prg = cl.Program(self.context, kern_src).build()


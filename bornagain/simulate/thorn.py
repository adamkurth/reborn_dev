import time
import pkg_resources
import sys
import os

import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array


import bornagain as ba
import clcore
import refdata


class ThornAgain:
    def __init__(self, q_vecs, atom_vecs, atomic_nums=None, load_default=True, group_size=32):
       
        if os.environ.get('BORNAGAIN_CL_GROUPSIZE') is not None:
            self.group_size = group_size
        else:
            self.group_size = int(os.environ.get('BORNAGAIN_CL_GROUPSIZE'))
        self.q_vecs = q_vecs.astype(np.float32)
        self.atom_vecs = atom_vecs.astype(np.float32)
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
            self.form_facts_arr = np.ones((self.Npix,1), dtype=np.float32)
            self.atomIDs = np.zeros(self.Nato)  # , dtype=np.int32)
            self.Nspecies = 1
            return

        croman_coef = refdata.get_cromermann_parameters(self.atomic_nums)
        form_facts_dict = refdata.get_cmann_form_factors(
            croman_coef, self.q_vecs)  # form factors!

        lookup = {}  # for matching atomID to atomic number
        self.form_facts_arr = np.zeros(
            (self.q_vecs.shape[0], len(form_facts_dict)), dtype=np.float32)
        for i, z in enumerate(form_facts_dict):
            lookup[z] = i  # set the key
            self.form_facts_arr[:,i] = form_facts_dict[z]
        self.atomIDs = np.array([lookup[z] for z in self.atomic_nums])
        
        self.Nspecies = np.unique( self.atomic_nums).size

        assert( self.Nspecies < 13) # can easily change this later if necessary... 
        #^ this assertion is so we can pass inputs to GPU as a float16, 3 q vectors and 13 atom species
        # where one is reserved to be a dummie 

    def _setup_openCL(self):
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(
            device_type=cl.device_type.GPU)

        self.context = cl.Context(devices=my_gpu_devices)
        self.queue = cl.CommandQueue(self.context)

    def load_program(self, which='default'):
        if which == 'default':
            self.prg = self.all_prg.qrf_default
            self.prg.set_scalar_arg_dtypes(
                [None, None, None, None, np.int32, np.int32])
            self._prime_buffers_default()

    def _prime_buffers_default(self):
        # allow these to overflow (not sure if necess.)
        self.Nextra_pix = (self.group_size - self.Npix % self.group_size)

        #mf = cl.mem_flags  # how to handle the buffers

#       make a rotation mat
        self.rot_mat = np.eye(3).ravel().astype(np.float32)

#       combine atom vectors and atom IDs (species IDs)
        self.atom_vecs = np.concatenate(
            (self.atom_vecs, self.atomIDs[:, None]), axis=1)
        
#       combine form factors with q_vectors for faster reads... 
        q_zeros = np.zeros((self.q_vecs.shape[0], 13))
        self.q_vecs = np.concatenate((self.q_vecs, q_zeros), axis=1)
        self.q_vecs[:,3:3+self.Nspecies] = self.form_facts_arr

#       make input buffers
        self.r_buff = clcore.to_device(
            self.atom_vecs, dtype=np.float32, queue=self.queue)
        self.q_buff = clcore.to_device( self.q_vecs, dtype=np.float32, queue=self.queue )
        self.rot_buff = clcore.to_device(
            self.rot_mat, dtype=np.float32, queue=self.queue)

#       make output buffer
        self.A_buff = clcore.to_device(
            np.zeros(self.Npix+self.Nextra_pix), dtype=np.complex64, queue=self.queue )

#       list of kernel args
        self.prg_args = [self.queue, (self.Npix+self.Nextra_pix,),(self.group_size,),
                         self.q_buff.data, self.r_buff.data,
                         self.rot_buff.data, self.A_buff.data, self.Npix, self.Nato]

    def _set_rand_rot(self):
        self.rot_buff = clcore.to_device(
            self.rot_mat, dtype=np.float32, queue=self.queue)
        self.prg_args[5] = self.rot_buff.data  # consider kwargs ?

    def run(self, rand_rot=False, force_rot_mat=None):
#       set the rotation
        if rand_rot:
            self.rot_mat = ba.utils.random_rotation_matrix().ravel().astype(np.float32)
        else:
            self.rot_mat = np.eye(3).ravel().astype(np.float32)

        if force_rot_mat is not None:
            self.rot_mat = force_rot_mat.astype(np.float32)

        self._set_rand_rot()

#       run the program
        t = time.time()
        self.prg(*self.prg_args)
        Amps = self.A_buff.get() [:-self.Nextra_pix]
        print ("Took %.4f sec to complete..." % float(time.time() - t))

        return Amps

    def _load_sources(self):
        clcore_file = pkg_resources.resource_filename(
            'bornagain.simulate', 'clcore.cpp')
        with open(clcore_file, 'r') as f:
            kern_src = f.read()
        self.all_prg = cl.Program(self.context, kern_src).build()

def test():
    natom = 100
    coors = np.random.random( (natom,3) )
    atomZ = np.ones(natom)
    D = ba.detector.SimpleDetector() 
    T = ThornAgain(D.Q, coors, atomZ, group_size=32)
    A = T.run(rand_rot=1)
    I = D.readout(A)

if __name__=="__main__":
    test()




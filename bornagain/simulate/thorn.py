from __future__ import (absolute_import, division, print_function, unicode_literals)

import time
import pkg_resources
import sys
import os

import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array


import bornagain as ba
from bornagain.simulate import clcore, refdata

nextPow2 = lambda x: int(2**(np.ceil(np.log2(x))))
prevPow2 = lambda x: int(2**(np.floor(np.log2(x))))

def make_q_vectors(qmin, qmax, wavelen, dq=0.02, dphi=0.05, pow2=None):
# ~~~~~~~ your code goes here(below) ~~~~~~~~

    Nphi = int(2.*np.pi / dphi)
    Nq = int( (qmax - qmin) / dq)

   
    if pow2 is not None: 
        assert (pow2 in ['prev','next'])
        if pow2 == 'prev':
            Nphi = prevPow2(Nphi)
            Nq = prevPow2(Nq)
        if pow2=='next':
            Nphi = nextPow2(Nphi)
            Nq = nextPow2(Nq)

    phi_values = np.arange(Nphi)*2.*np.pi / Nphi
    q_values = np.arange( Nq) * (qmax-qmin) / Nq

    img_sh = (Nq, Nphi)

    qs = np.array([q_values] * Nphi).T
    phis = np.array([phi_values] * Nq) 
    thetas = np.arcsin(qs * wavelen / 4. / np.pi)

    qx = np.cos(thetas) * qs * np.cos(phis)
    qy = np.cos(thetas) * qs * np.sin(phis)
    qz = np.sin(thetas) * qs

# ~~~~~~ your code ends here ~~~~~~~~

    qxyz = np.vstack((qx.ravel(), qy.ravel(), qz.ravel())).T
    print("\t\tsimulating into %d q vectors." % qxyz.shape[0])
    return img_sh, qxyz


class ThornAgain:
    def __init__(self, q_vecs, atom_vecs, atomic_nums=None, which='default', 
        group_size=32, cpu=False, sub_com=False):
        
        assert( which in ['default','kam'])
       
        if os.environ.get('BORNAGAIN_CL_GROUPSIZE') is None:
            self.group_size = group_size
        else:
            self.group_size = int(os.environ.get('BORNAGAIN_CL_GROUPSIZE'))
        
        self.q_vecs = q_vecs.astype(np.float32)
        self.atom_vecs = atom_vecs.astype(np.float32)
        self.atomic_nums = atomic_nums

        if sub_com:
            self.atom_vecs -= self.atom_vecs.mean(0)

        np.random.seed()

        self.cpu = cpu

#       set dimensions
        self.Nato = np.int32(atom_vecs.shape[0])
        self.Npix = np.int32(q_vecs.shape[0])
#       allow these to overflow (not sure if necess.)
        self.Nextra_pix = (self.group_size - self.Npix % self.group_size)

        self._make_croman_data()
        assert(self._setup_openCL())
        self._load_sources()

        self.which=which
        self.load_program(self.which)

        self.CLCORE = clcore.ClCore(queue=self.queue)
        
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
        if group_size > max_group_size:
            sys.stderr.write('Changing group size from %d to %d.\n'
                     'Set BORNAGAIN_CL_GROUPSIZE=%d to avoid this error.\n' 
                     % (group_size, max_group_size, max_group_size))
            group_size = max_group_size
        self.group_size = group_size
        
    

    def _make_croman_data(self):
        if self.atomic_nums is None:
            self.form_facts_arr = np.ones((self.Npix+self.Nextra_pix,1), dtype=np.float32)
            self.atomIDs = np.zeros(self.Nato)  # , dtype=np.int32)
            self.Nspecies = 1
            return

        croman_coef = refdata.get_cromermann_parameters(self.atomic_nums)
        form_facts_dict = refdata.get_cmann_form_factors(croman_coef, self.q_vecs)

        self.atom_id_lookup = {}  # for matching atomID to atomic number
        self.form_facts_arr = np.zeros(
            (self.Npix+self.Nextra_pix, len(form_facts_dict)), dtype=np.float32)
        for i, z in enumerate(form_facts_dict):
            self.atom_id_lookup[z] = i  # set the key
            self.form_facts_arr[:self.Npix,i] = form_facts_dict[z]
        self.atomIDs = np.array([self.atom_id_lookup[z] for z in self.atomic_nums])
        
        self.Nspecies = np.unique( self.atomic_nums).size

        assert( self.Nspecies < 13) # can easily change this later if necessary... 
        #^ this assertion is so we can pass inputs to GPU as a float16, 3 q vectors and 13 atom species
        # where one is reserved to be a dummie 

    def _setup_openCL(self, plat_ind=None):
        platforms = cl.get_platforms()
        if len(platforms) >1 and plat_ind is None:
            print("Provide a plat_ind to select compute platform")
            return False
        elif len(platforms) == 1:
            plat_ind = 0
        if self.cpu:
            device_type = cl.device_type.CPU
        else:
            device_type = cl.device_type.GPU

        my_devices = platforms[plat_ind].get_devices(
            device_type=device_type)
    
        print ("Using device:",my_devices[0] ) 

        if not my_devices:
            print("No devices selected")
            return False

        self.context = cl.Context(devices=my_devices)
        self.queue = cl.CommandQueue(self.context)
        return True

    def load_program(self, which):
        if which == 'default':
            self.prg = self.all_prg.qrf_default
            self.prg.set_scalar_arg_dtypes(
                [None, None, None, None,  np.int32])
            self._prime_buffers_default()
        elif which=='kam':
            self.prg = self.all_prg.qrf_kam
            self.prg.set_scalar_arg_dtypes(
                [None, None, None, None,None, np.int32])
            self._prime_buffers_kam()

    def _prime_buffers_kam(self):
        """ get all the data onto the GPU"""
#       setup device buffers
        self._set_atom_buffer()

        self._set_q_buffer()
        
        self._set_rot_buffer()

        self._set_com_buffer()
        
        self._set_amp_buffer()

        self._set_args_kam()

    def _prime_buffers_default(self):
        """ get all the data onto the GPU"""
#       setup device buffers
        self._set_atom_buffer()

        self._set_q_buffer()
        
        self._set_rot_buffer()

        self._set_amp_buffer()

        self._set_args()

    def _set_atom_buffer(self):
        """ combine atom vectors with atom ID
        to make a 4-vector for openCL device"""
#       combine atom vectors and atom IDs (species IDs)
        self.atom_vecs = np.concatenate(
            (self.atom_vecs, self.atomIDs[:, None]), axis=1)
        self.r_buff = clcore.ClCore.to_device_static(
            self.atom_vecs, dtype=np.float32, queue=self.queue)
    
    def _set_q_buffer(self):
        """ combine form factors and q-vectors
        for openCL device"""
#       combine form factors with q_vectors for faster reads... 
        q_zeros = np.zeros((self.Npix+self.Nextra_pix, 16))
        q_zeros[:self.Npix, :3] = self.q_vecs
        q_zeros[:,3:3+self.Nspecies] = self.form_facts_arr
        #self.q_vecs = np.concatenate((self.q_vecs, q_zeros), axis=1)
        #self.q_vecs[:,3:3+self.Nspecies] = self.form_facts_arr
        
        self.q_buff = clcore.ClCore.to_device_static( q_zeros, dtype=np.float32, queue=self.queue )

    def _set_com_buffer(self):
#       make a dummie translation vector
        self.com_vec = np.zeros(3).astype(np.float32)
        self.com_buff = clcore.ClCore.to_device_static(
            self.com_vec, dtype=np.float32, queue=self.queue)

    def _set_rot_buffer(self):
#       make a dummie rotation mat
        self.rot_mat = np.eye(3).ravel().astype(np.float32)
        self.rot_buff = clcore.ClCore.to_device_static(
            self.rot_mat, dtype=np.float32, queue=self.queue)
    def _set_amp_buffer(self):
#       make output buffer; initialize as 0s
        self.A_buff = clcore.ClCore.to_device_static(
            np.zeros(self.Npix+self.Nextra_pix), dtype=np.complex64, queue=self.queue )

#   list of kernel args
    def _set_args(self):
        self.prg_args = [self.queue, (self.Npix+self.Nextra_pix,),(self.group_size,),
                         self.q_buff.data, self.r_buff.data,
                         self.rot_buff.data, self.A_buff.data, self.Nato]
    
    def _set_args_kam(self):
        self.prg_args = [self.queue, (self.Npix+self.Nextra_pix,),(self.group_size,),
                         self.q_buff.data, self.r_buff.data,
                         self.rot_buff.data, self.com_buff, self.A_buff.data, self.Nato]

    def update_rbuff(self, new_atoms, new_z=None):
        """
        new_atoms, float Nx3 of atoms
        new_z, float Nx1 atomic numbers
        """
        na = new_atoms.shape[0]
        self.atom_vecs =  new_atoms
        if new_z is not None:
            assert( new_z.shape[0] == na)
            assert( np.all( [z in self.atom_id_lookup for z in set(new_z)]))
            self.atomIDs = np.array( [self.atom_id_lookup[z] for z in new_z])
        else:
            self.atomIDs = np.zeros(na)
        
        self.Nato = na
        self._set_atom_buffer()

        #self.r_buff = clcore.ClCore.to_device_static(
        #    self.atom_vecs, dtype=np.float32, queue=self.queue)
        self.prg_args[4] = self.r_buff.data
        self.prg_args[-1] = self.Nato

    def _set_rand_rot(self):
        self.rot_buff = clcore.ClCore.to_device_static(
            self.rot_mat, dtype=np.float32, queue=self.queue)
        self.prg_args[5] = self.rot_buff.data  # consider kwargs ?
    
    def _set_com_vec(self):
        self.com_buff = clcore.ClCore.to_device_static(
            self.com_vec, dtype=np.float32, queue=self.queue)
        self.prg_args[6] = self.com_buff.data 

    def run_transient(self, rand_rot=False, force_rot_mat=None, com=None):
        
#       set the rotation
        if rand_rot:
            self.rot_mat = ba.utils.random_rotation_matrix().ravel().astype(np.float32)
        else:
            self.rot_mat = np.eye(3).ravel().astype(np.float32)

        if force_rot_mat is not None:
            self.rot_mat = force_rot_mat.astype(np.float32)

        self._set_rand_rot()

        if com is not None:
            self.com_vec = com
            self._set_com_vec()

#       run the program
        self.prg(*self.prg_args)

    def run(self, rand_rot=False, force_rot_mat=None, com=None):
#       set the rotation
        if rand_rot:
            self.rot_mat = ba.utils.random_rotation_matrix().ravel().astype(np.float32)
        else:
            self.rot_mat = np.eye(3).ravel().astype(np.float32)

        if force_rot_mat is not None:
            self.rot_mat = force_rot_mat.astype(np.float32)

        self._set_rand_rot()

        if com is not None:
            self.com_vec = com
            self._set_com_vec()
    
#       run the program
        self.prg(*self.prg_args)

        Amps = self.release_amplitudes() 

        return Amps
        
    def release_amplitudes(self):
        Amps = self.A_buff.get() [:-self.Nextra_pix]
        
        self._set_amp_buffer()
        if self.which=='kam':
            self.prg_args[7]=self.A_buff.data
        else:
            self.prg_args[6]=self.A_buff.data

        return Amps

    def _load_sources(self):
        clcore_file = pkg_resources.resource_filename(
            'bornagain.simulate', 'clcore.cpp')
        
        with open(clcore_file, 'r') as f:
            kern_src = f.read()
        
        build_opts = ['-D', 'GROUP_SIZE=%d' % self.group_size]
        
        self.all_prg = cl.Program(self.context, kern_src).build(options=build_opts)

def test():
    natom = 1000
    coors = np.random.random( (natom,3) )
    atomZ = np.ones(natom)
    D = ba.detector.SimpleDetector(n_pixels=1000) 
    print ("\tSimulating into %d pixels"%D.Q.shape[0])
    T = ThornAgain(D.Q, coors, atomZ, cpu=False)
    A = T.run(rand_rot=1)
    I = D.readout(A)
    D.display()

    print("Passed testing mode!")

if __name__=="__main__":
    test()




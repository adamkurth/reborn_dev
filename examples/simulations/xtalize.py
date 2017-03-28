import time
import pkg_resources
import sys

import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array


import bornagain as ba
import bornagain.simulate.clcore as clcore
import bornagain.simulate.refdata as refdata
import bornagain.target.crystal as crystal


try:
    import matplotlib
except ImportError:
    pass


class SimpleDetector:
    def __init__(self, nPixels=2000, pixelSize=0.00005, detectorDistance=0.05, wavelength=1):

        self.pl = ba.detector.PanelList()

        self.nPixels = nPixels
        self.pixelSize=pixelSize
        self.detectorDistance=detectorDistance
        self.wavelength=wavelength

#       make a single panel detector:
        self.pl.simple_setup(nPixels, nPixels+1, pixelSize, detectorDistance, wavelength)
        self.q_vecs = self.pl[0].Q # q vectors!
        
        #self.cos_th = np.cos( np.arcsin( self.pl.stol*self.wavelength) )
        #self.sin_th = self.pl.stol* self.wavelength

        self.qmags = self.pl.Qmag 

#       shape of the 2D det panel (2D image)
        self.img_sh = (self.pl[0].nS, self.pl[0].nF)

    def det_info(self):
        print("num pixels: %d;  ")

    def readout(self, amplitudes):
        self.intens = (np.abs(amplitudes)**2).reshape(self.img_sh)
        return self.intens

    def readout_finite(self, amplitudes, qmin, qmax, flux=1e20):
        
        struct_fact =( np.abs( amplitudes)**2).astype(np.float64)

        if qmin < self.qmags.min():
            qmin = self.qmags.min()
        if qmax > self.qmags.max():
            qmax = self.qmags.max()

        ilow = np.where( self.qmags < qmin)[0]
        ihigh = np.where( self.qmags > qmax)[0]

        if ilow.size:
            struct_fact[ ilow] = 0
        if ihigh.size:
            struct_fact[ ihigh] = 0

        rad_electron = 2.82e-13 # cm
        phot_per_pix = struct_fact * self.pl.solidAngle * flux * rad_electron**2
        total_phot = int( phot_per_pix.sum())

        pvals = struct_fact / struct_fact.sum()

        self.intens = np.random.multinomial(  total_phot, pvals )
        
        self.intens = self.intens.reshape( self.img_sh)

        return self.intens

    def display(self):
        if 'matplotlib' not in sys.modules:
            print("You need matplotlib to plot!")
            return

        plt = matplotlib.pylab
        fig = plt.figure()
        ax = plt.gca()

        qx_min ,qy_min = self.q_vecs[:,:2].min(0)
        qx_max ,qy_max = self.q_vecs[:,:2].max(0)
        extent = (qx_min, qx_max, qy_min, qy_max)
        ax_img = ax.imshow( np.log1p(self.intens), extent=extent, cmap='viridis', interpolation='lanczos')
        
        ax.set_xlabel(r'$q_x\,\,\AA^{-1}$')
        ax.set_ylabel(r'$q_y\,\,\AA^{-1}$')
        
        cbar = fig.colorbar(ax_img)
        cbar.ax.set_ylabel('photon counts', rotation=270, labelpad=15)
            


class Molecule:
    def __init__(self, pdbFile = '/home/dermen/.local/ext/bornagain/examples/data/pdb/2LYZ.pdb'):
        self.cryst = crystal.structure(pdbFile)
        self.atom_vecs = self.cryst.r*1e10 # atom positions!
        self.Z = self.cryst.Z

class Lattice:
    def __init__(self, a=79.1, b=79.1, c=37.9 ):
#       unit cell edges
        self.a = np.array([a, 0, 0]) # angstroms
        self.b = np.array([0, b, 0])
        self.c = np.array([0, 0, c])

    def assemble(self, n_unit=10, spherical=False):

#       lattice coordinates
        self.vecs = np.array( [ i*self.a + j*self.b + k*self.c 
            for i in xrange( n_unit) 
                for j in xrange( n_unit) 
                    for k in xrange(n_unit) ] )

#       sphericalize the lattice.. 
        if spherical:
            self.vecs = ba.utils.sphericalize( self.vecs)


class ThornAgain:
    def __init__(self, q_vecs, atom_vecs, atomic_nums=None, load_default=True):
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


    def _set_rand_rot(self):
        self.rot_buff = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.rot_mat)
        self.prg_args[7] = self.rot_buff # consider kwargs ?

    def run(self, rand_rot=False, force_rot_mat=None):
#       set the rotation        
        if rand_rot:
            self.rot_mat = ba.utils.random_rotation_matrix().T.ravel().astype(np.float32)
        else:
            self.rot_mat = np.eye(3).ravel().astype(np.float32)
        
        if force_rot_mat is not None:
            self.rot_mat = force_rot_mat.astype(np.float32)

        self._set_rand_rot()

#       run the program
        t = time.time()
        self.prg( *self.prg_args) 
        print ("Took %.4f sec to run the kernel"%float(time.time()-t))

#       copy from device to host
        t = time.time()
        Amps = self.A_buff.get()[:-self.Nextra_pix]
        
        #a = self.Areal_buff.get()[:-self.Nextra_pix]
        #b = self.Aimag_buff.get()[:-self.Nextra_pix]
        #cl.enqueue_copy(self.queue, self.Areal, self.Areal_buff)
        #cl.enqueue_copy(self.queue, self.Aimag, self.Aimag_buff)
        print ("Took %.4f sec to copy data from device to host"%float(time.time()-t))

        return Amps


    def _load_sources(self):
        clcore_file = pkg_resources.resource_filename('bornagain.simulate','clcore.cpp')
        with open(clcore_file,'r') as f:
            kern_src = f.read()
        self.all_prg = cl.Program(self.context, kern_src).build()


if __name__ == '__main__':


    det = SimpleDetector()
    mol = Molecule()
    lat = Lattice()
    lat.assemble(spherical=True, n_unit=40)
    
#   simulators
    Tmol = ThornAgain(det.q_vecs, mol.atom_vecs, mol.Z )
    Tlat = ThornAgain(det.q_vecs, lat.vecs )
    
    Amol = Tmol.run(rand_rot=True)
    Alat = Tlat.run(force_rot_mat=Tmol.rot_mat)


#   get the intensity
    qmin = 0.2 # inv angstrom
    qmax = 5 
    flux = 1e26 # photon per cm**2
    img = det.readout_finite(Alat*Amol, qmin=qmin, qmax=qmax, flux=flux)
    det.display()














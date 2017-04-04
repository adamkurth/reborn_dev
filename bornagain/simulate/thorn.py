import time
import pkg_resources
import sys

import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array


import bornagain as ba
import clcore
import refdata
import bornagain.target.crystal as crystal


try:
    import matplotlib
except ImportError:
    pass


class SimpleDetector:
    def __init__(self, nPixels=1000, pixelSize=0.00005,
                 detectorDistance=0.05, wavelength=1):

        self.pl = ba.detector.PanelList()

        self.nPixels = nPixels
        self.pixelSize = pixelSize
        self.detectorDistance = detectorDistance
        self.wavelength = wavelength

#       make a single panel detector:
        self.pl.simple_setup(
            nPixels,
            nPixels + 1,
            pixelSize,
            detectorDistance,
            wavelength)
        self.q_vecs = self.pl[0].Q  # q vectors!

        #self.cos_th = np.cos( np.arcsin( self.pl.stol*self.wavelength) )
        #self.sin_th = self.pl.stol* self.wavelength

        self.qmags = self.pl.Qmag

#       shape of the 2D det panel (2D image)
        self.img_sh = (self.pl[0].nS, self.pl[0].nF)

    def readout(self, amplitudes):
        self.intens = (np.abs(amplitudes)**2).reshape(self.img_sh)
        return self.intens

    def readout_finite(self, amplitudes, qmin, qmax, flux=1e20):
        struct_fact = (np.abs(amplitudes)**2).astype(np.float64)

        if qmin < self.qmags.min():
            qmin = self.qmags.min()
        if qmax > self.qmags.max():
            qmax = self.qmags.max()

        ilow = np.where(self.qmags < qmin)[0]
        ihigh = np.where(self.qmags > qmax)[0]

        if ilow.size:
            struct_fact[ilow] = 0
        if ihigh.size:
            struct_fact[ihigh] = 0

        rad_electron = 2.82e-13  # cm
        phot_per_pix = struct_fact * self.pl.solidAngle * flux * rad_electron**2
        total_phot = int(phot_per_pix.sum())

        pvals = struct_fact / struct_fact.sum()

        self.intens = np.random.multinomial(total_phot, pvals)

        self.intens = self.intens.reshape(self.img_sh)

        return self.intens

    def display(self, use_log=True, vmax=None):
        if 'matplotlib' not in sys.modules:
            print("You need matplotlib to plot!")
            return

        plt = matplotlib.pylab
        fig = plt.figure()
        ax = plt.gca()

        qx_min, qy_min = self.q_vecs[:, :2].min(0)
        qx_max, qy_max = self.q_vecs[:, :2].max(0)
        extent = (qx_min, qx_max, qy_min, qy_max)
        if use_log:
            ax_img = ax.imshow(
                np.log1p(
                    self.intens),
                extent=extent,
                cmap='viridis',
                interpolation='lanczos')
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('log(photon counts)', rotation=270, labelpad=12)
        else:
            assert(vmax is not None)
            ax_img = ax.imshow(
                self.intens,
                extent=extent,
                cmap='viridis',
                interpolation='lanczos',
                vmax=vmax)
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('photon counts', rotation=270, labelpad=12)

        ax.set_xlabel(r'$q_x\,\,\AA^{-1}$')
        ax.set_ylabel(r'$q_y\,\,\AA^{-1}$')


class Atoms:
    def __init__(self, xyz, atomic_num, elem=None):
        self.xyz = xyz
        self.x = self.xyz[:, 0]
        self.y = self.xyz[:, 1]
        self.z = self.xyz[:, 2]
        self.Z = atomic_num

        self.coor = np.zeros((self.x.shape[0], 4))
        self.coor[:, :3] = self.xyz
        self.coor[:, 3] = self.Z

        self.coor[:, 3] = self.Z

        self.elem = elem
        if elem is not None:
            self.xyz_format = np.zeros((self.x.shape[0], 4), dtype='S16')
            self.xyz_format[:, 0] = self.elem
            self.xyz_format[:, 1:] = self.xyz.astype(str)

    @classmethod
    def aggregate(cls, atoms_list):

        xyz = np.vstack([a.xyz for a in atoms_list])

        if all([a.elem is not None for a in atoms_list]):
            elem = np.hstack([a.elem for a in atoms_list])
        else:
            elem = None

        Z = np.hstack([a.Z for a in atoms_list])

        return cls(xyz, Z, elem)

    def to_xyz(self, fname):
        if self.elem is not None:
            np.savetxt(fname, self.xyz_format, fmt='%s')
        else:
            print("Cannot save to xyz because element strings were not provided...")

    def set_elem(self, elem):
        """sets list of elements names for use in xyz format files"""
        elem = np.array(elem, dtype='S16')
        assert(self.elem.shape[0] == self.x.shape[0])
        self.elem = elem


class Molecule(crystal.structure):
    def __init__(self, *args, **kwargs):
        crystal.structure.__init__(self, *args, **kwargs)

        self.atom_vecs = self.r * 1e10  # atom positions!

        self.lat = Lattice(self.a * 1e10, self.b * 1e10, self.c * 1e10,
                           self.alpha * 180 / np.pi, self.beta * 180 / np.pi, self.gamma * 180 / np.pi)

        self.atom_fracs = self.mat_mult_many(self.Oinv * 1e-10, self.atom_vecs)

    def get_1d_coords(self):
        x, y, z = map(np.array, zip(*self.atom_vecs))
        return x, y, z

    def get_1d_frac_coords(self):
        x, y, z = map(np.array, zip(*self.atom_fracs))
        return x, y, z

    def mat_mult_many(self, M, V):
        """ helper for applying matrix multiplications on many vectors"""
        return np.einsum('ij,kj->ki', M, V)

    def transform(self, x, y, z):
        """x,y,z are fractional coordinates"""
        xyz = np.zeros((x.shape[0], 3))
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = z
        xyz = self.mat_mult_many(self.O * 1e10, xyz)
        return Atoms(xyz, self.Z, self.elements)

    def get_monomers(self):
        monomers = []
        for R, T in zip(self.symRs, self.symTs):
            transformed = self.mat_mult_many(R, self.atom_fracs) + T
            transformed = self.mat_mult_many(self.O * 1e10, transformed)
            monomers.append(Atoms(transformed, self.Z, self.elements))
        return monomers


class Lattice:
    def __init__(self, a=281., b=281., c=165.2,
                 alpha=90., beta=90., gamma=120.):
        """
        a,b,c are in Angstroms
        alpha, beta, gamma are in degrees
        default is for PS1
        """
#       unit cell edges
        alpha = alpha * np.pi / 180.
        beta = beta * np.pi / 180.
        gamma = gamma * np.pi / 180.

        cos = np.cos
        sin = np.sin
        self.V = a * b * c * np.sqrt(1 - cos(alpha)**2 - cos(beta) **
                                     2 - cos(gamma)**2 + 2 * cos(alpha) * cos(beta) * cos(gamma))
        self.a = np.array([a, 0, 0])
        self.b = np.array([b * cos(gamma), b * sin(gamma), 0])
        self.c = np.array([c * cos(beta),
                           c * (cos(alpha) - cos(beta) *
                                cos(gamma)) / sin(gamma),
                           self.V / (a * b * sin(gamma))])
        self.O = np.array([self.a, self.b, self.c]).T
        self.Oinv = np.linalg.inv(self.O)

    def assemble(self, n_unit=10, spherical=False):

        #       lattice coordinates
        self.vecs = np.array([i * self.a + j * self.b + k * self.c
                              for i in xrange(n_unit)
                              for j in xrange(n_unit)
                              for k in xrange(n_unit)])

#       sphericalize the lattice..
        if spherical:
            self.vecs = ba.utils.sphericalize(self.vecs)


class ThornAgain:
    def __init__(self, q_vecs, atom_vecs, atomic_nums=None, load_default=True):
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

    def _setup_openCL(self):
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(
            device_type=cl.device_type.GPU)

        self.group_size = clcore.group_size
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

        mf = cl.mem_flags  # how to handle the buffers

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
            self.queue, self.atom_vecs, dtype=np.float32)
        self.q_buff = clcore.to_device( self.queue, self.q_vecs, dtype=np.float32 )
        self.rot_buff = clcore.to_device(
            self.queue, self.rot_mat, dtype=np.float32)

#       make output buffer
        self.A_buff = clcore.to_device( self.queue, None,
            shape=(self.Npix+self.Nextra_pix), dtype=np.complex64 )

#       list of kernel args
        self.prg_args = [self.queue, (self.Npix,), None,
                         self.q_buff.data, self.r_buff.data,
                         self.rot_buff.data, self.A_buff.data, self.Npix, self.Nato]

    def _set_rand_rot(self):
        self.rot_buff = clcore.to_device(
            self.queue, self.rot_mat, dtype=np.float32)
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

#       copy from device to host
        Amps = self.A_buff.get()[:-self.Nextra_pix]
        #cl.enqueue_copy(self.queue, self.A, self.A_buff)
        print ("Took %.4f sec to complete..." % float(time.time() - t))

        return Amps

    def _load_sources(self):
        clcore_file = pkg_resources.resource_filename(
            'bornagain.simulate', 'clcore.cpp')
        with open(clcore_file, 'r') as f:
            kern_src = f.read()
        self.all_prg = cl.Program(self.context, kern_src).build()







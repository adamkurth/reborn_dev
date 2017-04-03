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
    def __init__(self, nPixels=1000, pixelSize=0.00005, detectorDistance=0.05, wavelength=1):

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

        self.intens = np.random.multinomial(total_phot, pvals)
        
        self.intens = self.intens.reshape( self.img_sh)

        return self.intens

    def display(self, use_log=True, vmax=None):
        if 'matplotlib' not in sys.modules:
            print("You need matplotlib to plot!")
            return

        plt = matplotlib.pylab
        fig = plt.figure()
        ax = plt.gca()

        qx_min ,qy_min = self.q_vecs[:,:2].min(0)
        qx_max ,qy_max = self.q_vecs[:,:2].max(0)
        extent = (qx_min, qx_max, qy_min, qy_max)
        if use_log:
            ax_img = ax.imshow( np.log1p(self.intens), extent=extent, cmap='viridis', interpolation='lanczos')
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('log(photon counts)', rotation=270, labelpad=12)
        else:
            assert( vmax is not None)
            ax_img = ax.imshow( self.intens, extent=extent, cmap='viridis', interpolation='lanczos', vmax=vmax)
            cbar = fig.colorbar(ax_img)
            cbar.ax.set_ylabel('photon counts', rotation=270, labelpad=12)

        ax.set_xlabel(r'$q_x\,\,\AA^{-1}$')
        ax.set_ylabel(r'$q_y\,\,\AA^{-1}$')
        
            
class Atoms:
    def __init__(self, xyz, atomic_num, elem=None):
        self.xyz = xyz
        self.x = self.xyz[:,0]
        self.y = self.xyz[:,1]
        self.z = self.xyz[:,2]
        self.Z = atomic_num

        self.coor = np.zeros((self.x.shape[0],4))
        self.coor[:,:3] = self.xyz
        self.coor[:,3] = self.Z
        
        self.coor[:,3] = self.Z

        self.elem = elem
        if elem is not None:
            self.xyz_format = np.zeros( (self.x.shape[0],4), dtype='S16')
            self.xyz_format[:,0] = self.elem
            self.xyz_format[:,1:] = self.xyz.astype(str)
   
    @classmethod
    def aggregate(cls, atoms_list):
        
        xyz = np.vstack(  [a.xyz for a in atoms_list])

        if all([ a.elem is not None for a in atoms_list]):
            elem = np.hstack(  [a.elem for a in atoms_list])
        else:
            elem = None

        Z =  np.hstack(  [a.Z for a in atoms_list])

        return cls(  xyz, Z, elem)

    def to_xyz(self, fname):
        if self.elem is not None:
            np.savetxt(fname, self.xyz_format, fmt='%s')
        else:
            print("Cannot save to xyz because element strings were not provided...")

    def set_elem(self, elem):
        """sets list of elements names for use in xyz format files"""
        elem = np.array(elem, dtype='S16')
        assert( self.elem.shape[0] == self.x.shape[0])
        self.elem = elem

class Molecule(crystal.structure):
    def __init__(self, *args, **kwargs):
        crystal.structure.__init__(self, *args, **kwargs)
        
        self.atom_vecs = self.r*1e10 # atom positions!

        self.lat = Lattice( self.a*1e10, self.b*1e10, self.c*1e10, 
            self.alpha*180/np.pi, self.beta*180/np.pi, self.gamma*180/np.pi)

        self.atom_fracs = self.mat_mult_many( self.Oinv*1e-10, self.atom_vecs)

    def get_1d_coords(self):
        x,y,z = map(np.array, zip(*self.atom_vecs))
        return x,y,z
    
    def get_1d_frac_coords(self):
        x,y,z = map(np.array, zip(*self.atom_fracs))
        return x,y,z

    def mat_mult_many(self,M,V):
        """ helper for applying matrix multiplications on many vectors"""
        return np.einsum('ij,kj->ki', M,V )

    def transform(self, x, y, z):
        """x,y,z are fractional coordinates"""
        xyz = np.zeros( (x.shape[0],3))
        xyz[:,0] = x
        xyz[:,1] = y
        xyz[:,2] = z
        xyz = self.mat_mult_many(self.O*1e10, xyz)
        return Atoms( xyz, self.Z, self.elements)

    def get_monomers(self):
        monomers = []
        for R,T in zip( self.symRs, self.symTs ):
            transformed = self.mat_mult_many(R, self.atom_fracs) + T
            transformed = self.mat_mult_many( self.O*1e10, transformed)
            monomers.append( Atoms( transformed, self.Z, self.elements))
        return monomers


class Lattice:
    def __init__(self, a=281., b=281., c=165.2, alpha=90., beta=90., gamma=120. ):
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
        self.V = a*b*c*np.sqrt(1-cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma)  )
        self.a = np.array([a,0,0])
        self.b = np.array([b*cos(gamma), b*sin(gamma),0])
        self.c = np.array([c*cos(beta), c*(cos(alpha)-cos(beta)*cos(gamma))/sin(gamma),  self.V/(a*b*sin(gamma))])
        self.O = np.array(  [self.a, self.b, self.c]).T
        self.Oinv = np.linalg.inv( self.O  ) 

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

#       inputs
        self.atomID_buff = clcore.to_device( self.queue, self.atomIDs, dtype=np.int32 )
        self.form_facts_buff = clcore.to_device( self.queue, self.form_facts_arr, dtype=np.float32 )
        self.r_buff = clcore.to_device( self.queue, self.atom_vecs, dtype=np.float32 )
        self.q_buff = clcore.to_device( self.queue, self.q_vecs, dtype=np.float32 )
        self.rot_buff = clcore.to_device( self.queue, self.rot_mat, dtype=np.float32 )

#       outputs 
        self.A_buff = clcore.to_device( self.queue, None, shape=(self.Npix+self.Nextra_pix), dtype=np.complex64 )
    
        self.prg_args = [ self.queue, (self.Npix,), None, self.q_buff.data, self.r_buff.data, self.form_facts_buff.data, 
            self.atomID_buff.data, self.rot_buff.data, self.A_buff.data, self.Npix, self.Nato ]

    def _set_rand_rot(self):
        #self.rot_buff = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.rot_mat)
        #self.prg_args[7] = self.rot_buff # consider kwargs ?
        self.rot_buff = clcore.to_device( self.queue, self.rot_mat, dtype=np.float32 )
        self.prg_args[7] = self.rot_buff.data # consider kwargs ?

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
        self.prg( *self.prg_args) 

#       copy from device to host
        Amps = self.A_buff.get()[:-self.Nextra_pix]
        
        #cl.enqueue_copy(self.queue, self.Areal, self.Areal_buff)
        #cl.enqueue_copy(self.queue, self.Aimag, self.Aimag_buff)
        print ("Took %.4f sec to complete..."%float(time.time()-t))

        return Amps

    def _load_sources(self):
        clcore_file = pkg_resources.resource_filename('bornagain.simulate','clcore.cpp')
        with open(clcore_file,'r') as f:
            kern_src = f.read()
        self.all_prg = cl.Program(self.context, kern_src).build()

if __name__ == '__main__':

#   simple detector object, reads in scattering amplitudes and outputs intensity
    det = SimpleDetector(nPixels=2000)
    
#   so.. lets generate some amplitudes...

#   made this class to inherit from bornagain.crystal.structure
#   (it uses Angstrom units)
    mol = Molecule(pdbFilePath='1jb0.pdb')
    
#   1d vectors of fractional coors, for use below
    x,y,z = mol.get_1d_frac_coords()

#   This is the funky bit only pertaining to the 
#   problem involving PhotoSys1
#   since the symOps dont generate trimer aggregats
#   we have to adjust them (hence the plus 1 business)...

#   Each call to mol.transform returns an Atoms instance
    mono1 = mol.transform( x, y, z ) # transforms are in fractional coors 
    mono2 = mol.transform( -y+1, x-y, z )
    mono3 = mol.transform( -x+y+1,-x+1, z )
    trimer1 = Atoms.aggregate([ mono1, mono2, mono3])
    #trimer1.to_xyz('trimer1.xyz') # save for vis with e.g. Pymol
    
#   make another trimer, offset doesn't matter, but orientation does.. 
    mono4 = mol.transform( -x+2,-y+1,z+.5 )
    mono5 = mol.transform( x-y+1,x,z+.5 )
    mono6 = mol.transform( y+1,-x+y+1,z+.5 )
    trimer2 = Atoms.aggregate([ mono4, mono5, mono6])
    #trimer2.to_xyz('trimer2.xyz')


#   if you want the full unit you can simply
    monomers = mol.get_monomers()
#   ... monomers contains, in this case, 6 Atoms instances
    hexemer = Atoms.aggregate(monomers) # e.g.

    # build the lattice, extend it n_units along each lattice vector 
    mol.lat.assemble(n_unit=100, spherical=1) # 100 x 100 x 100 Unit cells

#   lattice coordinates (in cartesian units) are
    lat_vecs = mol.lat.vecs # only after running assemble

#   simulators
    Tmol = ThornAgain(det.q_vecs, trimer1.xyz, trimer1.Z )
    Amol = Tmol.run(rand_rot=True)
    
    Tlat = ThornAgain(det.q_vecs, mol.lat.vecs )
    Alat = Tlat.run(force_rot_mat=Tmol.rot_mat)

#   get the intensity
    qmin = 0.2 # inv angstrom
    qmax = 5 
    flux = 1e27 # photon per cm**2
    img = det.readout_finite(Alat*Amol, qmin=qmin, qmax=qmax, flux=flux)
    #det.display()

#   that was for one trimer... 
#   now we want to simulate the scattering for two trimers
#   aranged on the lattice, with random occupancy 
#   occuring at the boundary

#   the easiest way to do this is to create a sub lattice (called Lattice2)
#   similar to the first (Lattice1), only shifted by the symmetry translation operator
#   ... then we can put the second trimer (trimer2 above) on this second lattice
#   ...then by making a "large" double lattice (one per trimer) we can simulate
#   random shape boundary effects by selecting different spherical
#   regions within the double lattice, as a randomly drawn sphere will isolate different
#   points from each lattice

#   first lets make a big lattice
    mol.lat.assemble(n_unit=40,spherical=False)
    lat1 = mol.lat

#   to create the second lattice, first examine the symmetry translations:
    print mol.symTs
    #[array([[ 0.,  0.,  0.]]),
    # array([[ 0. ,  0. ,  0.5]]),
    # array([[ 0.,  0.,  0.]]),
    # array([[ 0. ,  0. ,  0.5]]),
    # array([[ 0.,  0.,  0.]]),
    # array([[ 0. ,  0. ,  0.5]])]

#   as you can see, there are two, one trimer is at 0,0,0
#   and the other is at 0,0,.5

#   trimer 2 is placed
#   at z+.5 (corresponding to lattice vector c)
    lat1 = mol.lat.vecs
    lat2 = lat1 + mol.lat.c*.5
    dbl_lat = np.vstack( [lat1, lat2] )

#   these are the intices correspinding to 
#   each lattice
    n_lat_pts = mol.lat.vecs.shape[0]
    inds_lat1 = np.arange( n_lat_pts  )
    inds_lat2 = np.arange( n_lat_pts, 2*n_lat_pts  )

#   find the lattice center point
    lat_center = dbl_lat.mean(0)
#   and distance from center of each lattice point
    lat_rads = np.sqrt( np.sum( (dbl_lat - lat_center)**2,1))
    
#   maximum direction from center such that 
#   we can fit a sphere inside the lattice
    max_rad = min( dbl_lat.max(0))/2.

#   choose an inner sphere within which to jitter the
#   sphere center
    inner_rad = 0.2*max_rad

#   choose a random point within the inner sphere
    phi = np.random.uniform(0,2*pi)
    costheta = np.random.uniform(-1,1)
    u = random.random()

    theta = arccos( costheta )
    r = inner_rad * np.power( u, 1/3. )

    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )

    rand_sphere_cent = np.array( [x,y,z])

#   choose a random radius now, such that it does not exceed max_rad
    outer_rad = max_rad - inner_rad
    u = random.random()
    r = outer_rad * np.power( u, 1/3. )

#   select all points within the newly defined sphere
    lat_rads_rand = np.sqrt( np.sum( (dbl_lat - rand_sphere_cent)**2,1))

    rand_lat1 = lat1[ (lat_rads_rand < r) [inds_lat1]]
    rand_lat2 = lat2[ (lat_rads_rand < r)[inds_lat2]]

#   simulate each lattice now
    Tlat1 = ThornAgain(det.q_vecs, rand_lat1 )
    Tlat2 = ThornAgain(det.q_vecs, rand_lat2 )
    Alat1 = Tlat1.run(force_rot_mat=Tmol.rot_mat)
    Alat2 = Tlat2.run(force_rot_mat=Tmol.rot_mat)

#   now just simulate the second trimer, but relative the lattice2.. 
    Tmol2 = ThornAgain(det.q_vecs, trimer2.xyz-mol.lat.c*.5, trimer2.Z )
    Amol2 = Tmol.run(force_rot_mat=Tmol.rot_mat)
    

    


import os
import sys
import time
import numpy as np
import h5py
from glob import glob
import matplotlib.pyplot as plt
sys.path.append("../..")
import bornagain as ba
from bornagain.units import r_e, hc, keV
from bornagain.simulate.clcore import ClCore

################
# COPY PASTE THE CLASS FOR NOW
###############
from scipy.spatial import cKDTree, distance
class Place(cKDTree):
    def __init__(self, box_edge, min_dist, max_try=10000, *args, **kwargs):
        """
        Place points into a box of edge length box_edge, and don't let any two points
        
        Parameters
        ==========
        get within min_dist from one another.

        box_edge, float
            side length of the box to place spheres into
        min_dist, float
            minimum distance between two points in the box
        max_try, int
            number of times to try placing a new point such 
            that is does not overlap
        
        """
        np.random.seed()
        a = np.random.uniform( 0, box_edge, (1,3) )
        cKDTree.__init__(self, a, *args, **kwargs)
        self.min_dist = min_dist
        self.box_edge = box_edge
        self.max_try= max_try
        self.too_dense = False
 
    def insert(self):
        """adds a new point to the box"""
        new_pt = np.random.uniform( 0, self.box_edge, (1,3) )
        n_try=0
        is_overlapping = self.query( new_pt, distance_upper_bound=self.min_dist)[0] < np.inf # query for a nearest neighbor
        while is_overlapping:
            new_pt = np.random.uniform( 0, self.box_edge, (1,3) )
            is_overlapping = self.query( new_pt, distance_upper_bound=self.min_dist)[0] < np.inf
            n_try +=1
            if n_try > self.max_try:
                print("Getting too tight in here!")
                self.too_dense = True
                return
        data = np.concatenate(( self.data, new_pt)) # combine new pt and old pts
        super(Place, self).__init__(data) # re-initialize the parent class with new data
 
def place_spheres(Vf, sph_rad = 1., box_edge=None, Nspheres=1000, tol=0.01):
    """
    Vf, float
        Fraction of sample volume occupied by spheres
    Nspheres, int
        how many spheres in the sample volume
    tol, float
        minimum distance the unit spheres can be to one another
    """
#   volume of a unit sphere
    sph_vol = (4 / 3.) * np.pi * (sph_rad)**3
 
    if box_edge is not None:
#       then we let Nspheres be a free
        box_vol = box_edge**3
        Nspheres = int( (box_vol * Vf) / sph_vol )
    else:
#       then Nspheres determines the size of the box
        box_vol = sph_vol * Nspheres / Vf
        box_edge = np.power(box_vol, 1 / 3.)
 
    min_dist = 2*sph_rad+tol  # diameter plus tol,
 
    #print("Placing %d spheres into a box of side length %.4f"%(Nspheres, box_edge))
 
    p = Place(box_edge, min_dist) # init the Placer
    while p.n < Nspheres:
        p.insert() # insert pt!
        if p.too_dense: 
            print("\tbreaking insert loop with %d/%d spheres"%(p.n,Nspheres))
            break
     
    return p.data
################
# END COPY/ PASTE
################


n_molecules = 1000
photon_energy = 6.5 / keV
wavelength = hc/photon_energy

beam_vec = np.array([0, 0, 1.0])

# Single molecule atomic positions:
r = np.array([[0,     0, 0],
              [5e-10, 0, 0]])

# make the placer for putting molecules in 3D space
r_size =distance.pdist(r).max() 
placer = Place( box_size=100e-9, min_dist=r_size)
# for now the box is 100 nm x 100 nm x 100 nm

n_atoms = r.shape[0]
f = ba.simulate.atoms.get_scattering_factors([25]*r.shape[0], photon_energy)  # This number is irrelevant

pad = ba.detector.PADGeometry()
pad.simple_setup(n_pixels=100, pixel_size=100e-9, distance=1.0)
q = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)

clcore = ClCore(group_size=32)

# for now I will use CromMan because it works, I will add in Henke support asap
clcore.prime_cromermann_simulator(q, [6,6] )
q = core.get_q_cromermann()
r = core.get_r_cromermann(atom_pos, sub_com=False)

# For one shot, jiggle molecule positions and orientations and phases
rs = np.zeros((n_molecules*n_atoms,3))
fs = np.zeros((n_molecules*n_atoms), dtype=clcore.complex_t)

# insert all the molecules...
for n in range( 0, n_molecules):
    placer.insert()
mol_center_of_masses = placer.data

for n in range(0,n_molecules):
    
    mol_com = mol_center_of_masses[ n ]
    clore.run_cromermann(q, r, rand_rot=True, com=mol_com)
    
    # Rotate one molecule
    #R = ba.utils.random_rotation()
    # Add the rotated postions to list of atoms
    #rs[(n*n_atoms):((n+1)*n_atoms), :] = np.dot(R, r.T).T

    # No translation yet...

    # As a hack we will randomize the structure factor phases
    phases = np.random.random(f.shape)*2.0*np.pi
    fs[(n*n_atoms):((n+1)*n_atoms)] = f*np.exp(1j*phases)


A = clcore.phase_factor_pad(rs, fs, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength)

I = np.abs(A)**2  # As a practical limit, this intensity should reflect the fact that we get only one fluorescence
                # photon per atom

# Next: repeat many times, make lots of patterns, then take two-point correlations

# For incident fluence, we get the cross section (area) for photoabsorption.  We want one incdient photon per that area.

# Questions:
# How does SNR scale with number of molecules?
# How many shots needed?


# Something is wrong with this output!
imdisp = I.reshape(pad.shape())
plt.imshow(np.log10(imdisp+1e-5))
plt.show()

from __future__ import division

import os
import warnings
warnings.filterwarnings("ignore")

import sys
from time import time
import pylab as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # We expect a divide-by-zero, which is corrected... I don't like the annoying message...
from scipy.stats import binned_statistic_dd
from scipy.spatial import distance, cKDTree
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

sys.path.append("../..")
import bornagain as ba
from bornagain.target import crystal, Place
from bornagain.units import hc, keV
from bornagain.utils import vec_norm
from bornagain.simulate.clcore import ClCore
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import h5py

# Viewing choices
qtview = True

# save info
outdir = "1-1mol_1modes_25unit"
if not os.path.exists(outdir):
    os.makedirs( outdir)
file_stride = 500
save_kvecs = "k_vecs_xtal"
save_normfactor="norm_factor_xtal"
print_stride=100
# output file names:
out_pre = "1-1mol_1modes_25unit"
Waxs_file = os.path.join( outdir, "%s.Waxs"%out_pre)
Nshots_file = os.path.join( outdir, "%s.Nshots"%out_pre)

norm_factor = None
#norm_factor = np.load("norm_factor.npy")
# this norm factor should correspond to your k_vecs, set as None to create, but it takes some time... 

#output waxs pattern
qmax_waxs = 12 # inverse angstrom
Nq_waxs = 512

# How many diffraction patterns to simulate
n_patterns = 2000
Num_modes = 1

# Intensity of the fluoress
photons_per_atom = 1

# whether to use Henke or Cromer mann
use_henke = True  # if False, then use Cromer-mann version

# Information about the object
n_molecules = 1
box_size = 1000e-9
do_rotations = True #False
do_phases = True
do_translations = True

# Settings for pixel-array detector
n_pixels_per_dim = 100 # along a row
pixel_size = 0.001 # meters, (small pixels that we will bin average later)
detector_distance = .4 # meter

# Settings for spherical detector
spherical_detector = False #True
n_subdivisions = 3
radius = 1

####################################

# Information about the emission
photon_energy = 10.5 / keV
wavelength = hc / photon_energy # in meters
beam_vec = np.array([0, 0, 1.0]) # This shouldn't matter...

# Atomic positions of Mn atoms:
pdb_file = '../data/pdb/3wu2.pdb'
cryst = crystal.Molecule(pdb_file)
is_manga = cryst.Z==25
r = cryst.r[ is_manga]
r = r[:4] # take the first monomer in assymetric unit, 
cryst.lat.assemble(n_unit=5)
lattice = cryst.lat.vecs*1e-10
r = np.vstack([ r+l for l in lattice])
r -= r.mean(0)  # mean sub, I dunno it matters or not , but for rotations maybe...

n_atoms = r.shape[0]
# maximum distance spanned by the molecule:
r_size = distance.pdist(r).max()

print('Will simulate %d patterns' % (n_patterns))

if spherical_detector:
    detect_type = "SPHERE"
    print('Creating spherical detector...')
    ico = ba.detector.IcosphereGeometry(n_subdivisions=n_subdivisions, radius=radius)
    verts, faces, fcs = ico.compute_vertices_and_faces()
    n_faces = faces.shape[0]
    q = (2 * np.pi / wavelength) * fcs #(fcs - beam_vec)
    print('%d pixels' % (q.shape[0]))
else:
    detect_type='BOX'
    n = n_pixels_per_dim # shortcut
    p = pixel_size # shortcut
    
    pad = ba.detector.PADGeometry()
    pad.n_fs = n
    pad.n_ss = n
    pad.fs_vec = [p,0,0]
    pad.ss_vec = [0,p,0]
    pad.t_vec = [ (-n*p+p)*.5 , (-n*p+p)*.5, detector_distance]
    
    
    q = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
    pad_sh= pad.shape()
    print("Made the pad")
    #   combine the qs into a single vector...
    
    k_vecs = pad.position_vecs()
    k_vecs = vec_norm( k_vecs) * 2 * np.pi / wavelength
    
    q12 = distance.cdist( k_vecs, k_vecs).ravel() # pair q distances
    if save_kvecs is not None:
        np.save(os.path.join( outdir, save_kvecs), k_vecs)
    print("The pads cover the range %.4f to %.4f inverse angstrom"%(q12.min()*1e-10, q12.max()*1e-10))
    print("Making solid angles...")
    sangs = pad.solid_angles2()
    SA_frac = sangs.sum() / 4 / np.pi

Npix = k_vecs.shape[0]

print("Simulating intensities for %d pixels in the %s detector.." %(Npix, detect_type))

clcore = ClCore(group_size=1,double_precision=True)
q_dev = clcore.to_device(q)
seconds = 0
t0 = t=  time()

qbins = np.linspace( 0, qmax_waxs * 1e10, Nq_waxs+1)
if norm_factor is None:
    # make normalization factor
    # doing it this way to save on RAM
    norm_factor = np.zeros( Nq_waxs)
    for ik,kval in enumerate(k_vecs):
        kdists = distance.cdist( [kval], k_vecs )
        kdigs = np.digitize( kdists, qbins)-1
        norm_factor += np.bincount( kdigs.ravel(), minlength=Nq_waxs)
        if ik%print_stride==0:
            print ( "Making norm factor: %d pixels remain..."% ( len(k_vecs) - ik))
    
    if save_normfactor:
        np.save( os.path.join( outdir, save_normfactor ), norm_factor)

def sparse_idi(J):
    idx = np.where(J)[0]
    dists =  distance.cdist( k_vecs[idx], k_vecs[idx] )
    digs = np.digitize(dists, qbins) - 1
    Js = J[idx]
    weights = np.outer( Js, Js ) 
    H = np.bincount( digs.ravel(), minlength=Nq_waxs , weights=weights.ravel())
    return H

temp_waxs, temp_Nshots = [],[]
waxs = np.zeros(Nq_waxs)

for pattern_num in range(0, n_patterns):

    # Random positions for each molecule
    if do_translations:
        #Ts = np.random.random([n_molecules, 3])*box_size
        placer = Place( box_edge=box_size, min_dist=r_size)  # box_size and r_size need the same units... 
#       insert all the molecules...
        
        for i_n in range( n_molecules):
            #if i_n %100 == 0:
            #    print("placing %d / %d molecules in box" % ( i_n+1, n_molecules ) )
            placer.insert()
        Ts = placer.data # this is same units as r_size and box_size...
    else:
        Ts = np.zeros([n_molecules, 3])

    # Randomly rotate and translate the molecules
    rs = []
    for n in range(0, n_molecules):
        if do_rotations:
            R = ba.utils.random_rotation()
        else:
            R = np.eye(3)
        T = Ts[n, :]
        
        rs.append(np.dot(R, r.T).T + T)
    
    rs = np.array(rs).reshape([n_molecules*n_atoms, 3])  # miliseconds slow down
    total_atoms = rs.shape[0]

    # Compute intensities
    J = np.zeros( Npix)

    for _ in range( Num_modes):
        if do_phases:
            phases = np.random.random(n_atoms * n_molecules) * 2 * np.pi
        else:
            phases = np.zeros([n_atoms * n_molecules])
        fs = np.exp(1j * phases)
        A = clcore.phase_factor_qrf(q_dev, rs, fs)
        I = np.abs(A) ** 2
        if I.dtype==np.float32:
            I = I.astype(np.float64)
        N_photons_measured =  int( SA_frac * photons_per_atom * total_atoms / Num_modes)
        J += np.random.multinomial( N_photons_measured , I / I.sum() )
    
    h = sparse_idi(J)
    waxs += h
    
    if pattern_num % file_stride == 0:
        waxs_norm = waxs / norm_factor
        temp_waxs.append(waxs_norm / waxs_norm[0] )
        temp_Nshots.append(pattern_num)
        np.save( os.path.join( outdir, "temp_waxs_%d"%pattern_num) , waxs_norm / waxs_norm[0])
        np.save( os.path.join( outdir, "temp_Nshots_%d"%pattern_num) , pattern_num)
    
    dt = time()-t
    if np.floor(dt) >= 3:
        t = time()
        sys.stdout.write('Pattern %6d of %6d ; %3.0f%% ; %3.3g patterns/second\n' %
                         (pattern_num, n_patterns, 100*pattern_num/float(n_patterns),
                          pattern_num/(time() - t0)))

# last save point:
waxs_norm = waxs / norm_factor
temp_waxs.append(waxs_norm / waxs_norm[0] )
temp_Nshots.append(pattern_num)
np.save( os.path.join( outdir, "temp_waxs_%d"%pattern_num) , waxs_norm / waxs_norm[0])
np.save( os.path.join( outdir, "temp_Nshots_%d"%pattern_num) , pattern_num)

np.save(Nshots_file, temp_Nshots)
np.save(Waxs_file, temp_waxs)
print("Saved np binary files %s.npy and %s.npy " %(Nshots_file, Waxs_file))


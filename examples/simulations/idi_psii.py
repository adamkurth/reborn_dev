from __future__ import division


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
from bornagain.simulate.clcore import ClCore
import pyqtgraph.opengl as gl
import pyqtgraph as pg

# Viewing choices
qtview = True

# How many diffraction patterns to simulate
n_patterns = 100

# Intensity of the fluoress
add_noise = True
photons_per_atom = 1000

# whether to use Henke or Cromer mann
use_henke = True  # if False, then use Cromer-mann version

# Information about the object
n_molecules = 1
box_size = 1000e-9
do_rotations = True #False
do_phases = True
do_translations = True


# Settings for pixel-array detector
n_pixels = 100
pixel_size = 0.001 # meters, (small pixels that we will bin average later)
detector_distance = .05 # meter
#pix_size = 0.1 * sqrt( 2*rmax*rmax/(wavelength*wavelength) -1 ) / ( n_pixels/2.) # Im not sure if this works, but trying to guess the min pixel size needed for shannon sampling at the edge of the detector.. 

# Settings for spherical detector
spherical_detector = True
n_subdivisions = 3
radius = 1


####################################


# Information about the emission
photon_energy = 16.5 / keV
wavelength = hc / photon_energy # in meters
beam_vec = np.array([0, 0, 1.0]) # This shouldn't matter...

# Atomic positions of Mn atoms:
pdb_file = '../data/pdb/3wu2.pdb'
cryst = crystal.structure(pdb_file)
is_manga = cryst.Z==25
r = cryst.r[ is_manga]
r = r[:4] # take the first monomer in assymetric unit, 
r -= r.mean(0)  # mean sub, I dunno it matters or not , but for rotations maybe...
# dimer test???
#r = np.array([[0, 0, 0], [5e-10, 0, 0]])
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
    n = n_pixels # shortcut
    p = pixel_size # shortcut
    
    pad1 = ba.detector.PADGeometry()
    pad1.n_fs = n
    pad1.n_ss = n
    pad1.fs_vec = [0,0,p]
    pad1.ss_vec = [0,p,0]
    pad1.t_vec = [ -detector_distance,  (-n*p+p)*.5 , (-n*p+p)*.5  ]
    #pad.simple_setup(n_pixels=n_pixels, pixel_size=pixel_size, distance=detector_distance)
    pad2 = ba.detector.PADGeometry()
    pad2.n_fs = n
    pad2.n_ss = n
    pad2.fs_vec = [0,0,p]
    pad2.ss_vec = [0,p,0]
    pad2.t_vec = [ detector_distance,  (-n*p+p)*.5 , (-n*p+p)*.5  ]
    #pad.simple_setup(n_pixels=n_pixels, pixel_size=pixel_size, distance=detector_distance)
    
    pad3 = ba.detector.PADGeometry()
    pad3.n_fs = n
    pad3.n_ss = n
    pad3.fs_vec = [p,0,0]
    pad3.ss_vec = [0,p,0]
    pad3.t_vec = [ (-n*p+p)*.5 , (-n*p+p)*.5, detector_distance]
    
    #pad.simple_setup(n_pixels=n_pixels, pixel_size=pixel_size, distance=detector_distance)
    
    q1 = pad1.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
    q2 = pad2.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
    q3 = pad3.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
    pad1_sh= pad1.shape()
    pad2_sh= pad2.shape()
    pad3_sh= pad3.shape()
    print ("Making 3 PADS each with shape %d x %d"% ( pad1_sh[0], pad1_sh[1] ))
    #   combine the qs into a single vector...
    q = np.vstack( ( q1,q2,q3) )
    k_vecs =np.vstack( (pad1.position_vecs(), pad2.position_vecs(), pad3.position_vecs()) )
    k_vecs *= 2 * np.pi / wavelength
    q12 = distance.cdist( k_vecs, k_vecs).ravel() # pair q distances
    nbins=512 # number of q bins
    qbins = np.linspace( q12.min(), q12.max(), nbins+1) # these are the histogram bins.. 
    print("The pads cover the range %.4f to %.4f inverse angstrom"%(q12.min()*1e-10, q12.max()*1e-10))
    plt.hist( qbins*1e-10, bins=qbins*1e-10)
    plt.xlabel("inverse angstrom")
    plt.ylabel("bin count")
    plt.show()
n_q = q.shape[0]

print("Simulating intensities for %d pixels in the %s detector.." %(n_q, detect_type))
I_sum = np.zeros( n_q, dtype=np.float64) # store the intensities
II_sum = np.zeros( n_q*n_q, dtype=np.float64)  # stores the correlations of intensities

clcore = ClCore(group_size=1)
q_dev = clcore.to_device(q)
seconds = 0
t0 = t=  time()
for pattern_num in range(0, n_patterns):

    # Random phases for each atom
    if do_phases:
        phases = np.random.random(n_atoms * n_molecules) * 2 * np.pi
    else:
        phases = np.zeros([n_atoms * n_molecules])
    fs = np.exp(1j * phases)

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

    # Compute intensities
    A = clcore.phase_factor_qrf(q_dev, rs, fs)
    I = np.abs(A) ** 2

    dt = time()-t
    # print(time(), t0, np.floor(dt), seconds)
    if np.floor(dt) >= 3:
        t = time()
        sys.stdout.write('Pattern %6d of %6d ; %3.0f%% ; %3.3g patterns/second\n' %
                         (pattern_num, n_patterns, 100*pattern_num/float(n_patterns),
                          pattern_num/(time() - t0)))

    I *= photons_per_atom * n_atoms * n_molecules / np.sum(I.ravel())
    if add_noise: I = np.random.poisson(I)

    I_sum += I # summing the intensities

    #II_sum += np.multiply.outer(I, I).ravel() # here is summing the correlations of intensities.. 
    #print("computing correlation")
    II_sum += np.einsum( 'i,j->ij', I,I).ravel()
#    or 
    #II_sum += (I[:,None]*I[None,:] ).ravel()
#   prob best to do this on te GPU, it might be the bottleneck.. 

print('Post-processing...')

if not spherical_detector:
#   make magnitude of k1-k2 vectors for binning
    print("computing distance matrix for all pairs of k1,k2")
    qbin_count = np.histogram( q12, bins=qbins )[0]
    qbin_sums = np.histogram( q12, bins=qbins , weights=II_sum)[0]
    plt.plot( 1e-10* ( qbins[:-1]*.5 + qbins[1:]*.5 ), qbin_sums / qbin_count) 
    plt.show()

if spherical_detector:
#   can use braodcasting here:
    #qq = [np.subtract.outer(fcs[:, i], fcs[:, i]).ravel() for i in range(0, 3)]
    #qq = 2*np.pi/wavelength*np.ravel(qq).reshape([3, n_q**2]).T.copy()
    
    qq = np.vstack( fcs[:,None] - fcs[None,:] ) * 2 * np.pi / wavelength
    q_mags = np.sqrt(np.sum(q**2, axis=1))
    max_q = 4*np.pi/wavelength

    n_bins_3d = 51
    III, _, _ = binned_statistic_dd(sample=qq, values=II_sum, bins=(n_bins_3d,)*3, range=[(-max_q, max_q)]*3, statistic='sum')
    IIIc, _, _ = binned_statistic_dd(sample=qq, values=np.ones(II_sum.shape), bins=(n_bins_3d,)*3, range=[(-max_q, max_q)]*3, statistic='sum')
    mean_ = np.mean(I_sum/n_patterns)
    III = (III/IIIc - (mean_)**2)/mean_**2
    c = np.floor(n_bins_3d/2).astype(np.int)
    min_ = np.min(III[np.isfinite(III)])
    III[~np.isfinite(III)] = min_
    III[c, c, c] = min_

    if qtview:
        print('Displaying results...')

        im = pg.image(np.transpose(III, axes=(1, 0, 2)))
        im.setCurrentIndex(np.floor(n_bins_3d/2).astype(np.int))

        qa = pg.mkQApp()
        face_colors = np.ones([n_faces, 4])
        for i in range(0,3): face_colors[:, i] = (I / np.max(I))*0.95 + 0.05
        vw = gl.GLViewWidget()
        vw.show()
        md = gl.MeshData(vertexes=verts, faces=faces, faceColors=face_colors)
        mi = gl.GLMeshItem(meshdata=md, smooth=False) #, edgeColor=np.array([0.1, 0.1, 0.1])*255, drawEdges=True)
        vw.addItem(mi)
        qa.exec_()


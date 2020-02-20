from __future__ import division

import sys
from time import time

import numpy as np
np.seterr(divide='ignore', invalid='ignore') # We expect a divide-by-zero, which is corrected... I don't like the annoying message...
from scipy.stats import binned_statistic_dd
from scipy.spatial import distance, cKDTree
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

sys.path.append("../..")
import bornagain as ba
from bornagain.units import hc, keV
from bornagain.simulate.clcore import ClCore
import pyqtgraph.opengl as gl
import pyqtgraph as pg

# Viewing choices
qtview = True

# How many diffraction patterns to simulate
n_patterns = 100

# Intensity of the fluoress
add_noise = False
photons_per_atom = 1



# whether to use Henke or Cromer mann
use_henke = True  # if False, then use Cromer-mann version

# Information about the object
n_molecules = 1
box_size = 10e-9
do_rotations = False
do_phases = True
do_translations = True

# Information about the emission
photon_energy = 6.5 / keV
wavelength = hc / photon_energy
beam_vec = np.array([0, 0, 1.0]) # This shouldn't matter...

# Single molecule atomic positions:
r = np.array([[0, 0, 0],
              [5e-10, 0, 0]])


r -= r.mean(0)  # mean sub, I dunno it matters or not , but for rotations maybe...

# Settings for pixel-array detector
n_pixels = 100
pixel_size = 0.0005
detector_distance = .05

# Settings for spherical detector
spherical_detector = True
n_subdivisions = 3
radius = 1


print('Will simulate %d patterns' % (n_patterns))

# TODO: move this to detector?




###########
# TODO: move this into e.g. target
#############

class Place(cKDTree):
    def __init__(self, box_edge, min_dist, max_try=10000, *args, **kwargs):
        """
        Place points into a box of edge length box_edge, and don't let any two points
        
        Parameters
        ==========
        get with    t0 = time()in min_dist from one another.

        box_edge, float
            side length of the box to place spheres into
        min_dist, float
            minimum distance between two points in the box
        max_try, int
            number of times to try placing a new point such 
            that is does not overlap
        
        """
        np.random.seed()
        a = np.random.uniform(0, box_edge, (1, 3))
        cKDTree.__init__(self, a, *args, **kwargs)
        self.min_dist = min_dist
        self.box_edge = box_edge
        self.max_try = max_try
        self.too_dense = False

    def insert(self):
        """adds a new point to the box"""
        new_pt = np.random.uniform(0, self.box_edge, (1, 3))
        n_try = 0
        # @dermen - why is the inequality comparing with inf?
        is_overlapping = self.query(new_pt, distance_upper_bound=self.min_dist)[
                             0] < np.inf  # query for a nearest neighbor
        while is_overlapping:
            new_pt = np.random.uniform(0, self.box_edge, (1, 3))
            is_overlapping = self.query(new_pt, distance_upper_bound=self.min_dist)[0] < np.inf
            n_try += 1
            if n_try > self.max_try:
                print("Getting too tight in here!")
                self.too_dense = True
                return
        data = np.concatenate((self.data, new_pt))  # combine new pt and old pts
        super(Place, self).__init__(data)  # re-initialize the parent class with new data




# def place_spheres(Vf, sph_rad=1., box_edge=None, Nspheres=1000, tol=0.01):
#     """
#     Vf, float
#         Fraction of sample volume occupied by spheres
#     Nspheres, int
#         how many spheres in the sample volume
#     tol, float
#         minimum distance the unit spheres can be to one another
#     """
#     #   volume of a unit sphere
#     sph_vol = (4 / 3.) * np.pi * (sph_rad) ** 3
#
#     if box_edge is not None:
#         #       then we let Nspheres be a free
#         box_vol = box_edge ** 3
#         Nspheres = int((box_vol * Vf) / sph_vol)
#     else:
#         #       then Nspheres determines the size of the box
#         box_vol = sph_vol * Nspheres / Vf
#         box_edge = np.power(box_vol, 1 / 3.)
#
#     min_dist = 2 * sph_rad + tol  # diameter plus tol,
#
#     print("Placing %d spheres into a box of side length %.4f" % (Nspheres, box_edge))
#
#     p = Place(box_edge, min_dist)  # init the Placer
#     while p.n < Nspheres:
#         p.insert()  # insert pt!
#         if p.too_dense:
#             print("\tbreaking insert loop with %d/%d spheres" % (p.n, Nspheres))
#             break
#
#     return p.data


###########
# END COPY/PASTE
###########


# maximum distance spanned by the molecule:
r_size = distance.pdist(r).max()
n_atoms = r.shape[0]
sphereical_detector = False
if spherical_detector:
    print('Creating spherical detector...')
    ico = ba.detector.IcosphereGeometry(n_subdivisions=n_subdivisions, radius=radius)
    verts, faces, fcs = ico.compute_vertices_and_faces()
    n_faces = faces.shape[0]
    q = (2 * np.pi / wavelength) * (fcs - beam_vec)
    print('%d pixels' % (q.shape[0]))
else:
    pad = ba.detector.PADGeometry()
    pad.simple_setup(n_pixels=n_pixels, pixel_size=pixel_size, distance=detector_distance)
    q = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)

n_q = q.shape[0]

# # Place molecules in a 10 nm box
# dimer_placer = Place(box_edge=10e-9, min_dist=r_size)
# for i in range(n_molecules):
#     dimer_placer.insert()
# dimer_pos = dimer_placer.data

print('Simulating intensities...')
I_sum = 0
II_sum = 0
clcore = ClCore(group_size=1)
q_dev = clcore.to_device(q)
seconds = 0
t0 = time()
t = time()
for pattern_num in range(0, n_patterns):

    if use_henke:

        # Random phases for each atom
        if do_phases:
            phases = np.random.random(n_atoms * n_molecules) * 2 * np.pi
        else:
            phases = np.zeros([n_atoms * n_molecules])
        fs = np.exp(1j * phases)

        # Random positions for each molecule
        if do_translations:
            Ts = np.random.random([n_molecules, 3])*box_size
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
            sys.stdout.write('Pattern %6d of %6d ; %3.0f%% ; %3.3g patterns/second\n' % (pattern_num, n_patterns, 100*pattern_num/float(n_patterns), pattern_num/(time() - t0)))

    else:  # use cromer/mann
        ##### alternatively, using this old code i wrote we dont have to make copies of molecule
        # init CLCORE
        pass
    #     clcore = ClCore(group_size=1)
    #     clcore.prime_cromermann_simulator(q, np.array([79., 79.]))  # put in two bogus atomic numbers for Carbon
    #     qcm = clcore.get_q_cromermann()
    #     rcm = clcore.get_r_cromermann(r, sub_com=1)  # takes atom positions of single molecule
    #     for n in range(n_molecules):
    #         #   make the random phases, pass to run_crommer_mann function
    #         phases = np.random.random(n_atoms) * 2 * np.pi
    #         #   run cromermann
    #         clcore.run_cromermann(qcm, rcm, rand_rot=True, com=dimer_pos[n], rand_phase=phases)
    #     A = clcore.release_amplitudes()
    #     I = np.abs(A) ** 2
    #
    I *= photons_per_atom * n_atoms * n_molecules / np.sum(I.ravel())
    if add_noise: I = np.random.poisson(I)

    I_sum += I
    II_sum += np.multiply.outer(I, I).ravel()

print('Post-processing...')

# I_sum /= n_patterns
# II_sum /= n_patterns

n_q = q.shape[0]
qq = [np.subtract.outer(fcs[:, i], fcs[:, i]).ravel() for i in range(0, 3)]
qq = 2*np.pi/wavelength*np.ravel(qq).reshape([3, n_q**2]).T.copy()
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
III[c,c,c] = min_

if qtview:
    print('Displaying results...')

    im = pg.image(np.transpose(III, axes=(1, 0, 2)))
    im.setCurrentIndex(np.floor(n_bins_3d/2).astype(np.int))

    qa = pg.mkQApp()
    face_colors = np.ones([n_faces, 4])
    for i in range(0,3): face_colors[:, i] = (I / np.max(I))*0.95 + 0.05
    vw = gl.GLViewWidget()
    vw.show()
    print(verts.shape)
    print(faces.shape)
    print(face_colors.shape)
    md = gl.MeshData(vertexes=verts, faces=faces, faceColors=face_colors)
    mi = gl.GLMeshItem(meshdata=md, smooth=False) #, edgeColor=np.array([0.1, 0.1, 0.1])*255, drawEdges=True)
    vw.addItem(mi)
    qa.exec_()

# glw = pg.GraphicsLayoutWidget()
# glw.addItem(vw)


# if spherical_detector:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     mesh = Poly3DCollection(verts[faces], facecolors=mpl.cm.gray(I_noisy / np.max(I_noisy)), alpha=1, linewidths=0)
#     ax.add_collection3d(mesh, )
#     ax.scatter(fcs[:, 0], fcs[:, 1], fcs[:, 2], c='g', s=0, lw=0, alpha=0)
#     ax.set_aspect('equal', adjustable='box')
#     plt.show()
# else:
#     imdisp = I_noisy.reshape(pad.shape())
#     plt.imshow(np.log10(imdisp + 1), interpolation='nearest', cmap='gray')
#     plt.show()


    # plt.figure().add_subplot(111, projection='3d').scatter(v[:,0], v[:,1], v[:,2])
    # plt.figure().add_subplot(111, projection='3d').plot_surface(v[:,0], v[:,1], v[:,2])
    # plt.show()


    # Questions:
    # How does SNR scale with number of molecules?
    # How many shots needed?

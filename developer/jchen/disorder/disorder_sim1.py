"""
Script to simulate diffraction patterns from disordered materials

Features:

To do:

Date Created: 14 Oct 2020
Last Modified: 28 Nov 2021
Author: Rick Kirian, Joe Chen
"""

from time import time
import numpy as np
# import pyqtgraph as pg
from reborn import detector
from reborn import source
from reborn.utils import trilinear_insert, vec_mag, vec_norm
from reborn.simulate.clcore import ClCore
# from reborn.simulate.examples import psi_pdb_file, lysozyme_pdb_file
from reborn.target import crystal, density
from reborn.viewers.qtviews import Scatter3D, bright_colors, colors, PADView
#from reborn.external.pyqtgraph.extras import keep_open
import scipy.constants as const

from scipy.spatial.transform import Rotation

eV = const.value('electron volt')
r_e = const.value("classical electron radius")


CMAP = 'viridis'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=6) 


#=================================================================================
# Program parameters

run_number = 3
N_pattern = 2
save_interval = 500

viewcrystal = False
addfacets = True

photon_energy = 1.8e3*eV  #50e3 * eV #1.8e3 * eV#50000*eV
pulse_energy = 1e-3
beam_diameter = 1e-6
pixel_size = 300e-6
detector_distance = 0.05#0.141884685 #10
n_pixels = 128 #500

#h_max = 25
#h_min = -25
h_corner_min = np.array([-15,-15,-10])
h_corner_max = np.array([15,15,10])
os = 4
n_h_bins = ((h_corner_max-h_corner_min)*os+1) * np.ones([3])


#=================================================================================
def random_rotation():
    """ Returns a random rotation matrix """
    return Rotation.random().as_matrix()


#=================================================================================
# Load the pdb file
# pdb_file = '2lyz.pdb' #'1jb0.pdb' #lysozyme_pdb_file
pdb_file = '1jb0.pdb'

cryst = crystal.CrystalStructure(pdb_file, tight_packing=True)
uc = cryst.unitcell
sg = cryst.spacegroup

cryst = crystal.CrystalStructure(pdb_file)
cryst.spacegroup.sym_rotations = cryst.spacegroup.sym_rotations[:]  # TODO: fix this
cryst.spacegroup.sym_translations = cryst.spacegroup.sym_translations[:]  # TODO: fix this
cryst.fractional_coordinates =  cryst.fractional_coordinates[:] # np.array([[0.4, 0, 0], [0.5, 0, 0]])  # TODO: fix this
spacegroup = cryst.spacegroup
unitcell = cryst.unitcell
print('# symmetry operations: %d' % (spacegroup.n_operations,))
print(unitcell)

# Coordinates of asymmetric unit in crystal basis x
au_x_vecs = cryst.fractional_coordinates
# Redefine spacegroup operators so that molecules COMs are tightly packed in unit cell
au_x_com = np.mean(au_x_vecs, axis=0)
mol_x_coms = np.zeros((spacegroup.n_molecules, 3))
mol_r_coms = np.zeros((spacegroup.n_molecules, 3))
for i in range(spacegroup.n_molecules):
    com = spacegroup.apply_symmetry_operation(i, au_x_com)
    spacegroup.sym_translations[i] -= com - (com % 1)
    mol_x_coms[i, :] = spacegroup.apply_symmetry_operation(i, au_x_com)
    mol_r_coms[i, :] = unitcell.x2r(spacegroup.apply_symmetry_operation(i, au_x_com))

# Setup beam and detector
beam = source.Beam(photon_energy=photon_energy, pulse_energy=pulse_energy, diameter_fwhm=beam_diameter)
pad = detector.PADGeometry(pixel_size=pixel_size, distance=detector_distance, n_pixels=n_pixels)
q_vecs = pad.q_vecs(beam=beam).copy()
q_max = np.max(pad.q_mags(beam=beam))
resolution = 2*np.pi/q_max
print('Resolution: %.3g A' % (resolution*1e10,))

mask = pad.beamstop_mask(beam=beam, q_min=2*np.pi/500e-10)
V = pad.position_vecs()
SA = pad.solid_angles()
P = pad.polarization_factors(beam=beam)
J = beam.photon_number_fluence
scale = pad.reshape(r_e ** 2 * SA * P * J)

# Density map for merging
dmap = crystal.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=10)
dens = np.zeros(dmap.shape)
denswt = np.zeros(dmap.shape)

# Atomic scattering factors
f = cryst.molecule.get_scattering_factors(beam=beam)

# Construct the finite lattice generators
max_lattice_size_in_each_dir = 11#41  # Make sure we have an odd value... for making hexagonal prisms
lats = [crystal.FiniteLattice(max_size=max_lattice_size_in_each_dir, unitcell=unitcell) for i in range(spacegroup.n_molecules)]

clcore = ClCore()
print('Computing with:', clcore.context.devices)
amps_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_mol_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
amps_lat_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
q_vecs_gpu = clcore.to_device(q_vecs, dtype=clcore.real_t)
q_rot_gpu = clcore.to_device(shape=q_vecs.shape, dtype=clcore.real_t)
f_gpu = clcore.to_device(f, dtype=clcore.complex_t)


# Put all the atomic coordinates on the gpu
mol_r_vecs_gpu = []
for i in range(spacegroup.n_molecules):
    mol_vecs = spacegroup.apply_symmetry_operation(i, au_x_vecs)
    mol_vecs = unitcell.x2r(mol_vecs)
    mol_r_vecs_gpu.append(clcore.to_device(mol_vecs, dtype=clcore.real_t))


#=================================================================================

mask2 = np.ones(pad.shape())

merge_sum = 0
weight_sum = 0
intensities_sum = 0

if viewcrystal:
    scat = Scatter3D()

print('Generating patterns')
for c in range(N_pattern):
    print(f'Generating pattern {c+1}')
    rot = random_rotation()
    trans = np.zeros(3)
    A = np.dot(rot, unitcell.a_mat)

    # Construct a finite lattice in the form of a hexagonal prism
    print('Adding Facets')
    # width = 7 + np.random.rand(1)*3
    # length = 20 + np.random.rand(1)*3
    width = 2+ np.random.rand(1)*3 #7 + np.random.rand(1)*3
    length = 3+ np.random.rand(1)*3 #20 + np.random.rand(1)*3

    if addfacets:
        for i in range(spacegroup.n_molecules):
            lat = lats[i]
            com = mol_x_coms[i, :]
            lat.add_facet(plane=[-1, 1, 0], length=width, shift=com)
            lat.add_facet(plane=[1, -1, 0], length=width, shift=com)
            lat.add_facet(plane=[1, 0, 0], length=width, shift=com)
            lat.add_facet(plane=[0, 1, 0], length=width, shift=com)
            lat.add_facet(plane=[-1, 0, 0], length=width, shift=com)
            lat.add_facet(plane=[0, -1, 0], length=width, shift=com)

            lat.add_facet(plane=[0, 0, 1], length=length, shift=com)
            lat.add_facet(plane=[0, 0, -1], length=length, shift=com)
    
    

    t = time()
    print('Rotating q vectors')
    clcore.rotate_translate_vectors(rot, trans, q_vecs_gpu, q_rot_gpu)
    

    print('Calculating crystal complex amplitudes')
    amps_gpu *= 0
    for i in range(spacegroup.n_molecules):
        print(f'here1 {i}')
        lat_vecs = lats[i].occupied_x_coordinates # + mol_x_coms[i, :]
        lat_vecs = unitcell.x2r(lat_vecs)
        clcore.phase_factor_qrf(q_rot_gpu, lat_vecs, a=amps_lat_gpu, add=False)
        # mol_vecs = spacegroup.apply_symmetry_operation(i, au_x_vecs)
        # mol_vecs = unitcell.x2r(mol_vecs)
        print(f'here2 {i}')
        clcore.phase_factor_qrf(q_rot_gpu, mol_r_vecs_gpu[i], f_gpu, a=amps_mol_gpu, add=False)
        print(f'here3 {i}')
        amps_gpu += amps_lat_gpu * amps_mol_gpu

        yay
        
        if viewcrystal:

            plt.figure()
            ax = plt.axes(projection='3d')
            xdata_tot = np.array([])
            ydata_tot = np.array([])
            zdata_tot = np.array([])
            for i in range(spacegroup.n_molecules):
                print(i)
                xdata = lat_vecs[:,0] + mol_r_coms[i, :][0]
                ydata = lat_vecs[:,1] + mol_r_coms[i, :][1]
                zdata = lat_vecs[:,2] + mol_r_coms[i, :][2]

                xdata_tot = np.concatenate([xdata_tot, xdata])
                ydata_tot = np.concatenate([ydata_tot, ydata])
                zdata_tot = np.concatenate([zdata_tot, zdata])

            ax.scatter3D(xdata_tot, ydata_tot, zdata_tot)
            plt.show(block=False)

            scat.add_points(lat_vecs + mol_r_coms[i, :], color=bright_colors(i, alpha=0.5), size=5) # Lattice positions
            scat.add_points(mol_vecs, color=bright_colors(i, alpha=0.5), size=1) # atom positions of the asymmetric unit molecule
    
    if viewcrystal:
        scat.add_rgb_axis()
        scat.add_unit_cell(cell=unitcell)
        scat.set_orthographic_projection()
        scat.show()

    # Diffracted intensities
    intensities = scale * np.abs(amps_gpu.get()) ** 2

    # Add Poisson noise to the intensities
    # intensities = np.random.poisson(intensities).astype(np.float64)

    print('Merging into 3D Fourier volume')
    h_vecs = unitcell.q2h(q_rot_gpu.get())#/2/np.pi)
    # print(h_vecs.shape, intensities.shape, mask2.shape)
    merge, weight = trilinear_insert(h_vecs, intensities.ravel(), h_corner_min, h_corner_max, n_h_bins, mask2.ravel())
    # density.trilinear_insertion(densities=dens, weights=denswt, vectors=unitcell.q2h(q_rot_gpu.get()/np.pi/2.),
    #                             vals=intensities.ravel(), corners=dmap.h_limits[:, 0].copy(),
    #                             deltas=(dmap.h_limits[:, 1]-dmap.h_limits[:, 0]))
    merge_sum += merge
    weight_sum += weight



    # intensities_sum += intensities



    print('Pattern %d: %.3f seconds' % (c+1, time()-t))

    if ((c+1) % save_interval) == 0:
        # if c+1 == save_interval:
        #     print('saving extra data')
        #     np.savez('run%04d_extras.npz' % (run_number,), q_corner_min=q_corner_min, q_corner_max=q_corner_max, n_q_bins=n_q_bins)
        print('saving checkpoint on pattern %d run %d' % (c+1, run_number))
        np.savez('run%04d_merge.npz' % (run_number), merge_sum=merge_sum, weight_sum=weight_sum,
                 h_corner_min=h_corner_min, h_corner_max=h_corner_max, n_h_bins=n_h_bins, num_patterns_merged = c+1)



    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(121)
    im = ax.imshow(intensities, clim=[0,500],cmap=CMAP)
    colorbar(ax, im)
    ax.set_title("diffracted intensity")
    ax = fig.add_subplot(122)
    im = ax.imshow(np.log10(intensities), cmap=CMAP)
    colorbar(ax, im)
    ax.set_title("log scale")

    # plt.suptitle("I_merge, numPattern=%d" % (c+1), fontsize=12)

    plt.show(block=False)


#yay

#dat = intensities*mask
#padview = PADView(pad_geometry=[pad], raw_data=[dat])
#padview.show_coordinate_axes()
#padview.set_levels(0, np.percentile(dat, 90))
#padview.show()

w = np.where(weight_sum > 0)[0]
merge_avg = merge_sum.copy()
merge_avg.flat[w] /= weight_sum.flat[w]
# pg.image(np.log10(merge_avg))
# print('Keeping open...')
# keep_open()







from skimage.measure import marching_cubes_lewiner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iso_val = 200.0
verts, faces, _, _ = marching_cubes_lewiner(x_best_upsamp, iso_val, spacing=(1, 1, 1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2])#, cmap='Spectral',
                #lw=1)
plt.show(block=False)




#=================================================================================
# Plot results



#weightout[weightout == 0] = 1
#I_merge /= weightout

# merge_avg = np.log10(merge_avg)

N_bin_cent = (np.array(merge_avg.shape)/2).astype(int)#np.round(n_h_bins / 2).astype(np.int)
clim_max = 10 #30000000

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(131)
im = ax.imshow(merge_avg[N_bin_cent[0],:,:], clim=[0,clim_max], cmap=CMAP)
colorbar(ax, im)
ax.set_title("qx-qy")
ax = fig.add_subplot(132)
im = ax.imshow(merge_avg[:,N_bin_cent[1],:], clim=[0,clim_max], cmap=CMAP)
colorbar(ax, im)
ax.set_title("qx-qz")
ax = fig.add_subplot(133)
im = ax.imshow(merge_avg[:,:,N_bin_cent[2]], clim=[0, clim_max], cmap=CMAP)
colorbar(ax, im)
ax.set_title("qy-qz")

plt.suptitle("I_merge, numPattern=%d" % (c+1), fontsize=12)

plt.show(block=False)







# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(131)
# im = ax.imshow(intensities_sum[N_bin_cent[0],:,:], clim=[0,clim_max], interpolation='nearest', cmap=CMAP)
# colorbar(ax, im)
# ax.set_title("qx-qy")
# ax = fig.add_subplot(132)
# im = ax.imshow(intensities_sum[:,N_bin_cent[1],:], clim=[0,clim_max], interpolation='nearest', cmap=CMAP)
# colorbar(ax, im)
# ax.set_title("qx-qz")
# ax = fig.add_subplot(133)
# im = ax.imshow(intensities_sum[:,:,N_bin_cent[2]], clim=[0, clim_max], interpolation='nearest', cmap=CMAP)
# colorbar(ax, im)
# ax.set_title("qy-qz")

# plt.suptitle("I_merge, numPattern=%d" % (c+1), fontsize=12)

# plt.show(block=False)

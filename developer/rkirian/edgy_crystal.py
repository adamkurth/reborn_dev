from time import time
import numpy as np
import pyqtgraph as pg
from bornagain import detector
from bornagain import source
from bornagain.utils import trilinear_insert, random_rotation, vec_mag, vec_norm
from bornagain.simulate.clcore import ClCore
from bornagain.simulate.examples import psi_pdb_file, lysozyme_pdb_file
from bornagain.target import crystal, density
from bornagain.viewers.qtviews import Scatter3D, bright_colors, colors, PADView
from bornagain.external.pyqtgraph.extras import keep_open
import scipy.constants as const
eV = const.value('electron volt')
r_e = const.value("classical electron radius")

run_number = 1
save_interval = 10
viewcrystal = True 
addfacets = True
photon_energy = 50000*eV
pulse_energy = 1e-3
beam_diameter = 1e-6
pixel_size = 300e-6
detector_distance = 10
n_pixels = 500

# Load up the pdb file
cryst = crystal.CrystalStructure(psi_pdb_file)
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
max_size = 41  # Make sure we have an odd value... for making hexagonal prisms
lats = [crystal.FiniteLattice(max_size=max_size, unitcell=unitcell) for i in range(spacegroup.n_molecules)]

clcore = ClCore()
print('Computing with:', clcore.context.devices)
amps_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
amps_mol_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
amps_lat_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)
q_vecs_gpu = clcore.to_device(q_vecs, dtype=clcore.real_t)
q_rot_gpu = clcore.to_device(shape=q_vecs.shape, dtype=clcore.real_t)
f_gpu = clcore.to_device(f, dtype=clcore.complex_t)

merge_sum = 0
weight_sum = 0

h_max = 30
h_min = -30
n_h_bins = (h_max-h_min)*4*np.ones([3])+1
h_corner_min = h_min*np.ones([3])
h_corner_max = h_max*np.ones([3])

mask2 = np.ones(pad.shape())

# Put all the atomic coordinates on the gpu
mol_r_vecs_gpu = []
for i in range(spacegroup.n_molecules):
    mol_vecs = spacegroup.apply_symmetry_operation(i, au_x_vecs)
    mol_vecs = unitcell.x2r(mol_vecs)
    mol_r_vecs_gpu.append(clcore.to_device(mol_vecs, dtype=clcore.real_t))



scat = None

for c in range(30):
    rot = random_rotation()
    trans = np.zeros(3)
    A = np.dot(rot, unitcell.a_mat)
    # Construct a finite lattice in the form of a hexagonal prism
    width = 7 + np.random.rand(1)*3
    length = 20 + np.random.rand(1)*3
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
    if viewcrystal:
        if scat != None:
            del scat
            
        scat = Scatter3D()



    t = time()
    clcore.rotate_translate_vectors(rot, trans, q_vecs_gpu, q_rot_gpu)
    amps_gpu *= 0
    for i in range(spacegroup.n_molecules):
        lat_vecs = lats[i].occupied_x_coordinates # + mol_x_coms[i, :]
        lat_vecs = unitcell.x2r(lat_vecs)
        clcore.phase_factor_qrf(q_rot_gpu, lat_vecs, a=amps_lat_gpu, add=False)
        # mol_vecs = spacegroup.apply_symmetry_operation(i, au_x_vecs)
        # mol_vecs = unitcell.x2r(mol_vecs)
        clcore.phase_factor_qrf(q_rot_gpu, mol_r_vecs_gpu[i], f_gpu, a=amps_mol_gpu, add=False)
        amps_gpu += amps_lat_gpu * amps_mol_gpu
        if viewcrystal:
            scat.add_points(lat_vecs + mol_r_coms[i, :], color=bright_colors(i, alpha=0.5), size=5)
            scat.add_points(mol_vecs, color=bright_colors(i, alpha=0.5), size=1)
    if viewcrystal:
        scat.add_rgb_axis()
        scat.add_unit_cell(cell=unitcell)
        scat.set_orthographic_projection()
        scat.show()
    intensities = scale * np.abs(amps_gpu.get()) ** 2
    # intensities = np.random.poisson(intensities).astype(np.float64)
    h_vecs = unitcell.q2h(q_rot_gpu.get()/2/np.pi)
    # print(h_vecs.shape, intensities.shape, mask2.shape)
    merge, weight = trilinear_insert(h_vecs, intensities.ravel(), h_corner_min, h_corner_max, n_h_bins, mask2.ravel())
    # density.trilinear_insertion(densities=dens, weights=denswt, vectors=unitcell.q2h(q_rot_gpu.get()/np.pi/2.),
    #                             vals=intensities.ravel(), corners=dmap.h_limits[:, 0].copy(),
    #                             deltas=(dmap.h_limits[:, 1]-dmap.h_limits[:, 0]))
    merge_sum += merge
    weight_sum += weight
    print('Pattern %d; %.3f seconds' % (c, time()-t,))
    if ((c+1) % save_interval) == 0:
        # if c+1 == save_interval:
        #     print('saving extra data')
        #     np.savez('run%04d_extras.npz' % (run_number,), q_corner_min=q_corner_min, q_corner_max=q_corner_max, n_q_bins=n_q_bins)
        print('saving checkpoint on pattern %d run %d' % (c+1, run_number))
        np.savez('run%04d_checkpoint%06d.npz' % (run_number, c+1), merge_sum=merge_sum, weight_sum=weight_sum,
                 h_corner_min=h_corner_min, h_corner_max=h_corner_max, n_h_bins=n_h_bins)

dat = intensities*mask
padview = PADView(pad_geometry=[pad], raw_data=[dat])
padview.show_coordinate_axes()
padview.set_levels(0, np.percentile(dat, 90))
padview.show()

w = np.where(weight_sum > 0)[0]
merge_avg = merge_sum.copy()
merge_avg.flat[w] /= weight_sum.flat[w]
pg.image(merge_avg)
print('Keeping open...')
keep_open()

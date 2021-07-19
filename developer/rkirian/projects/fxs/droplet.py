import numpy as np
from reborn import detector, utils
from reborn.source import Beam
from reborn.target import crystal, atoms
from reborn.simulate.form_factors import sphere_form_factor
import pyqtgraph as pg
from reborn.viewers.qtviews import view_pad_data
import scipy.constants as const
from numpy.fft import fftn, ifftn, fftshift
from reborn.simulate.clcore import ClCore
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# np.random.seed(0)  # Make random numbers that are reproducible
eV = const.value('electron volt')
r_e = const.value('classical electron radius')
NA = const.value('Avogadro constant')
h = const.h
c = const.c
water_density = 1000  # SI units, like everything else in reborn!
photon_energy = 9000*eV
wavelength = h*c/photon_energy
detector_distance = 2.4
pulse_energy = 0.2*4e-3
drop_radius = 70e-9/2
beam_diameter = 0.5e-6
d = 0.2e-9  # Minimum resolution
s = 4       # Oversampling factor
cell_size = 200e-10  # Unit cell size (assume P1, cubic)
pdb_file = '2LYZ'  # '1PCQ' '2LYZ' '1SS8' 'BDNA25_sp.pdb'

clcore = ClCore(double_precision=False, group_size=32)
beam = Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=pulse_energy)
fluence = beam.photon_number_fluence
pads = detector.cspad_2x2_pad_geometry_list(detector_distance=detector_distance)
f_dens = atoms.xraylib_scattering_density('H2O', water_density, photon_energy, approximate=True)

q_mags = pads.q_mags(beam=beam)
solid_angles = pads.solid_angles()
polarization_factors = pads.polarization_factors(beam=beam)
amps = r_e * f_dens * sphere_form_factor(radius=drop_radius, q_mags=q_mags)
I_sphere = np.abs(amps)**2*solid_angles*polarization_factors*fluence
I_sphere = np.random.poisson(I_sphere)  # Add some Poisson noise

print('Loading pdb file (%s)' % pdb_file)
uc = crystal.UnitCell(cell_size, cell_size, cell_size, np.pi/2, np.pi/2, np.pi/2)
sg = crystal.SpaceGroup('P1', [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])], [np.zeros(3)])
cryst = crystal.CrystalStructure(pdb_file, spacegroup=sg, unitcell=uc)
print('Setting up 3D mesh')
dmap = crystal.CrystalDensityMap(cryst, d, s)
print('Getting scattering factors')
f = cryst.molecule.get_scattering_factors(beam=beam)
print('Grid size: (%d, %d, %d)' % tuple(dmap.shape))
h = dmap.h_vecs  # Miller indices (fractional)

print('Creating density map directly from atoms')
x = cryst.unitcell.r2x(cryst.molecule.get_centered_coordinates())
rho = dmap.place_atoms_in_map(x, f, mode='nearest')
voxel_vol = dmap.voxel_volume
print('Water scatter (electron) density', f_dens)
w = np.where(rho != 0)
rho_prot = np.sum(rho[w])/(len(w[0])*dmap.voxel_volume)
solvent = f_dens*voxel_vol
rho[rho == 0] = solvent
rho -= solvent
F = fftn(rho)
F = fftshift(F)
I = np.abs(F)**2
rho_cell = fftshift(rho)
F_gpu = clcore.to_device(F)
q_vecs_gpu = clcore.to_device(pads.q_vecs(beam=beam))
amps = clcore.to_device(shape=pads.n_pixels, dtype=clcore.complex_t)
q_min = 2*np.pi*np.dot(dmap.h_min, cryst.unitcell.a_mat.T)
q_max = 2*np.pi*np.dot(dmap.h_max, cryst.unitcell.a_mat.T)
R = Rotation.random().as_matrix()
clcore.mesh_interpolation(F_gpu, q_vecs_gpu, N=dmap.shape, q_min=q_min, q_max=q_max, R=R, U=None, a=amps, add=False)

I_prot = np.abs(amps.get())**2 * r_e**2 * solid_angles * polarization_factors * fluence
# I_prot = np.random.poisson(I_prot)
print('# photons:', np.sum(I_prot))

# Direct atomistic simulation
I_protd = []
for pad in pads:
    I_protd.append(clcore.phase_factor_pad(r=cryst.molecule.coordinates, f=f, pad=pad, beam=beam, add=False, R=R))
I_protd = np.abs(np.concatenate(I_protd))**2


print('Display results')
# view_pad_data(pad_data=np.log10(I_sphere+1), pad_geometry=pads, show=True)
view_pad_data(pad_data=np.log10(I_prot+1), pad_geometry=pads, show=True)
# view_pad_data(pad_data=np.log10(I_protd+1), pad_geometry=pads, show=True)
# dispim = np.abs(rho_cell).astype(np.float64)
# dispim /= np.abs(f_dens*voxel_vol)
# dispim = np.sum(dispim, axis=0)
# pg.image(dispim)
if 1:
    fig = plt.figure()
    fig.add_subplot(2, 3, 1)
    I = np.abs(I).astype(np.float64)
    rho = np.abs(rho).astype(np.float64)
    dispim = np.log10(I[np.floor(dmap.shape[0] / 2).astype(np.int), :, :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 4)
    dispim = np.sum(rho, axis=0)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 2)
    dispim = np.log10(I[:, np.floor(dmap.shape[1] / 2).astype(np.int), :] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 5)
    dispim = np.sum(rho, axis=1)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 3)
    dispim = np.log10(I[:, :, np.floor(dmap.shape[2] / 2).astype(np.int)] + 10)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
    fig.add_subplot(2, 3, 6)
    dispim = np.sum(rho, axis=2)
    plt.imshow(dispim, interpolation='nearest', cmap='gray')
plt.show()
pg.mkQApp().exec_()

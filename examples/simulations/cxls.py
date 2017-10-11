#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import h5py
from glob import glob

sys.path.append("../..")
import bornagain as ba
from bornagain.units import r_e, hc, keV
import bornagain.simulate.clcore as core

# Lysozyme at CXLS
#pdbFile = '../data/pdb/1jb0.pdb'
pdbFile = '../data/pdb/2LYZ-P1.pdb'
do_monte_carlo = True
n_monte_carlo_iterations = 1000
n_pixels = 1000
pixel_size = 110e-6
detector_distance = 100e-3
photon_energy = 12.0 / keV
#pulse_energy = 0.0024
wavelength = hc / photon_energy
wavelength_fwhm = wavelength * 0.02
beam_diameter = 20e-6
beam_divergence_fwhm = 0.010*(8e-6/beam_diameter)
transmission = 0.2
n_photons = 1e8 #pulse_energy / photon_energy  # 1e12
I0 = transmission * n_photons / (np.pi * (beam_diameter / 2.0) ** 2)
crystal_size = beam_diameter
mosaic_domain_size = 200e-9
mosaicity_fwhm = 0.0001
results_dir = '/data/temp/cxls-lysozyme-02/'
write_hdf5 = False
write_geom = False
cromer_mann = False
quiet = False
random_rotation = True
approximate_lattice_transform = True
rotation_axis = [1,0,0]
rotation_angle = 0.1
add_noise = False
overlay_wigner_cells = False
mask_direct_beam = True

# John gets ~100 photons per integrated peak with Lysozyme |F| ~ 5000

if not do_monte_carlo:
    n_monte_carlo_iterations = 1

if not quiet:
    write = sys.stdout.write
else:
    write = lambda x: x

section = '='*70+'\n'

write(section)
write('PDB file: %s\n' % (os.path.basename(pdbFile)))
write('Photons per pulse: %g\n' % (n_photons))
write('Photon energy: %g keV\n' % (photon_energy*keV))
write('Beam divergence: %g mrad FWHM\n' % (beam_divergence_fwhm*1e3))
write('Beam diameter: %g microns tophat\n' % (beam_diameter*1e6))
write('Spectral width: %g%% FWHM dlambda/lambda\n' % (100*wavelength_fwhm/wavelength))
write('Crystal size: %g microns\n' % (crystal_size*1e6))
write('Crystal mosaicity: %g radian FWHM\n' % (mosaicity_fwhm))
write('Crystal mosaic domain size: %g microns\n' % (mosaic_domain_size*1e6))
write(section)

# Things we probably don't want to think about
cl_group_size = 32
cl_double_precision = False

# Some overrides at the command prompt:
view = False
qtview = False
skview = False
if 'view' in sys.argv:
    import matplotlib.pyplot as plt
    view = True
if 'qtview' in sys.argv:
    import pyqtgraph as pg
    qtview = True
if 'skview' in sys.argv:
    from skimage.viewer import ImageViewer
    skview = True
if 'double' in sys.argv:
    cl_double_precision = True
if 'approximate' in sys.argv:
    approximate_lattice_transform = True
if 'save' in sys.argv:
    write_hdf5 = True
    write_geom = True

# Setup simulation engine
write('Setting up simulation engine... ')
clcore = core.ClCore(group_size=cl_group_size, double_precision=cl_double_precision)
write('done\n')

write('Will run %d Monte Carlo iterations\n' % (n_monte_carlo_iterations))

# Setup detector geometry
write('Configuring detector... ')
panel_list = ba.detector.PanelList()
panel_list.simple_setup(n_pixels, n_pixels, pixel_size, detector_distance, wavelength)
p = panel_list[0]
q = p.Q
qmag = p.Qmag
sa = p.solid_angle
P = p.polarization_factor
write('done\n')

# Get atomic coordinates and scattering factors from pdb file
write('Getting atomic coordinates and scattering factors... ')
cryst = ba.target.crystal.structure(pdbFile)
r = cryst.r
f = ba.simulate.atoms.get_scattering_factors(cryst.Z, ba.units.hc / wavelength)
write('done\n')
write('%d atoms per unit cell\n' % (len(f)))

# Make a mask for the direct beam
direct_beam_mask = np.ones((p.nS*p.nF))
direct_beam_mask[qmag < (2*np.pi/np.max(np.array([cryst.a,cryst.b,cryst.c])))] = 0

# Determine number of unit cells in whole crystal and mosaic domains
n_cells_whole_crystal = np.ceil(crystal_size / np.array([cryst.a, cryst.b, cryst.c]))
n_cells_mosaic_domain = np.ceil(mosaic_domain_size / np.array([cryst.a, cryst.b, cryst.c]))

# Setup function for lattice transform calculations
if approximate_lattice_transform:
    write('Using approximate (Gaussian) lattice transform\n')
    lattice_transform = clcore.gaussian_lattice_transform_intensities_pad
else:
    write('Using idealized (parallelepiped) lattice transform\n')
    lattice_transform = clcore.lattice_transform_intensities_pad

if write_geom:
    geom_file = results_dir + 'geom.geom'
    write('Writing geometry file %s\n' % geom_file)
    fid = open(geom_file, 'w')
    fid.write("photon_energy = %g\n" % (photon_energy * ba.units.eV))
    fid.write("clen = %g\n" % detector_distance)
    fid.write("res = %g\n" % (1 / pixel_size))
    fid.write("adu_per_eV = %g\n" % (1.0 / (photon_energy * ba.units.eV)))
    fid.write("0/min_ss = 0\n")
    fid.write("0/max_ss = %d\n" % (n_pixels - 1))
    fid.write("0/min_fs = 0\n")
    fid.write("0/max_fs = %d\n" % (n_pixels - 1))
    fid.write("0/corner_x = %g\n" % (-n_pixels / 2.0))
    fid.write("0/corner_y = %g\n" % (-n_pixels / 2.0))
    fid.write("0/fs = x\n")
    fid.write("0/ss = y\n")
    fid.close()

# Allocate memory on GPU device
write('Allocating GPU device memory... ')
r_dev = clcore.to_device(r, dtype=clcore.real_t)
f_dev = clcore.to_device(f, dtype=clcore.complex_t)
F_dev = clcore.to_device(np.zeros([p.nS*p.nF], dtype=clcore.complex_t))
S2_dev = clcore.to_device(shape=(p.nF, p.nS), dtype=clcore.real_t)
write('done\n')

R = ba.utils.rotation_about_axis(rotation_angle, rotation_axis)
if random_rotation: R = ba.utils.random_rotation()

if not cromer_mann:
    write('Simulating molecular transform from Henke tables... ')
    t = time.time()
    clcore.phase_factor_pad(r_dev,f_dev,p.T,p.F,p.S,p.beam.B,p.nF,p.nS,p.beam.wavelength,R,F_dev,add=False)
    F2 = np.abs(F_dev.get()) ** 2
    tf = time.time() - t
    write('%g s\n' % (tf))
else:
    write('Simulating molecular transform with cromer mann... ')
    t = time.time()
    clcore.prime_cromermann_simulator(q.copy(), cryst.Z.copy())
    q_cm = clcore.get_q_cromermann()
    r_cm = clcore.get_r_cromermann(r.copy(), sub_com=False)
    clcore.run_cromermann(q_cm, r_cm, rand_rot=False, force_rot_mat=R)
    A = clcore.release_amplitudes()
    F2 = np.abs(A) ** 2
    tf = time.time() - t
    write('%g s\n' % (tf))

abc = cryst.O.T.copy()
# S2 = np.zeros((p.nF, p.nS))
S2_dev *= 0

write('Simulating lattice transform... ')
time.sleep(0.001)
message = ''
tt = time.time()
for n in np.arange(1, (n_monte_carlo_iterations + 1)):

    t = time.time()
    if do_monte_carlo:
        B = ba.utils.random_beam_vector(beam_divergence_fwhm)
        w = np.random.normal(wavelength, wavelength_fwhm / 2.354820045, [1])[0]
        Rm = ba.utils.random_mosaic_rotation(mosaicity_fwhm).dot(R)
        T = p.T.copy() + p.F * (np.random.random([1]) - 0.5) + p.S * (np.random.random([1]) - 0.5)
    else:
        B = p.beam.B
        w = p.beam.wavelength
        Rm = R
        T = p.T
    lattice_transform(abc, n_cells_mosaic_domain, T, p.F, p.S, B, p.nF, p.nS, w, Rm, S2_dev, add=True)

    tf = time.time() - t
    if (n % 1000) == 0:
        write('\b' * len(message))
        message = '%3.0f%% (%5d; %7.03f ms)' % (n / float(n_monte_carlo_iterations) * 100, n, tf * 1e3)
        write(message)
write('\b' * len(message))
write('%g s                \n' % (time.time()-tt))

# Average the lattice transforms over MC iterations
S2 = S2_dev.get().ravel() / n
# Convert into useful photon units
I = I0 * r_e ** 2 * F2 * S2 * sa * P
# Scale up according to mosaic domain
n_domains = np.prod(n_cells_whole_crystal) / np.prod(n_cells_mosaic_domain)
I_ideal = I.copy()*n_domains
I_ideal = I_ideal.astype(np.float32)
I_noisy = np.random.poisson(I_ideal).astype(np.float32)

if write_hdf5:
    n_patterns = len(glob(results_dir + 'pattern-*.h5'))
    file_name = results_dir + 'pattern-%06d.h5' % (n_patterns + 1)
    write('Writing file %s\n' % file_name)
    fid = h5py.File(file_name, 'w')
    fid['/data/ideal'] = I_ideal.astype(np.float32).reshape((p.nS,p.nF))
    fid['/data/noisy'] = I_noisy.astype(np.int32).reshape((p.nS,p.nF))
    fid.close()

F2mean = np.mean(F2.flat[direct_beam_mask > 0])
Ncells = np.prod(n_cells_whole_crystal) 
darwin = I0*r_e**2*Ncells*F2mean*(wavelength**3/cryst.V)

write(section)
write('Simulation results, excluding direct-beam [000] reflection:\n')
write('Mean unit cell transform |F|^2: %g\n' % (F2mean))
write('Min pixel intensity: %g photons\n' % (np.min(I_ideal)))
write('Max pixel intensity: %g photons\n' % (np.max(I_ideal.flat[direct_beam_mask > 0])))
write('Total photon counts: %g\n' % (np.sum(I_ideal.flat[direct_beam_mask > 0])))
write(section)
write("Prediction from Darwin's formula:\n")
write('I0 = %g\n' % (I0))
write('r_e^2 = %g\n' % (r_e**2))
write('Ncells = %g\n' % (Ncells))
write('<|F|^2> = %g\n' % (F2mean))
write('lambda = %g\n' % (wavelength))
write('Vcell = %g\n' % (cryst.V))
write('------->>>>> I*re^2*<|F|^2>*(lambda/Vcell)^3 = %g <<<<<<-------\n' % (darwin))
write(section)
write("\n\nDone!\n\n")

# The following is just for display purposes:
if add_noise:
    write('Displaying with noise\n')
    I_display = I_noisy.copy()
else:
    write('Displaying **without** noise\n')
    I_display = I_ideal.copy()

if overlay_wigner_cells:
	hkl = cryst.O.dot(R).dot(q.T)/2.0/np.pi #.dot(cryst.Oinv)
	delta = hkl - np.round(hkl)
	delta = np.sqrt(np.sum(delta**2,axis=0))
	peak_mask = np.zeros((p.nF*p.nS))
	peak_mask[delta < 0.5] = 1
	I_display += peak_mask*1e-2

if mask_direct_beam:
   write('Masking direct beam\n')
   I_display *= direct_beam_mask

write('Taking log10 of simulated intensities\n')
I_display = np.log10(I_display + 1e-20)
I_display = I_display.reshape((p.nS, p.nF))

if qtview:
    img = pg.image(I_display, autoLevels=False, levels=[-0.4,np.log10(10)],
                   title='log10(I); %.1f micron; %.2f mrad; %.1f%% BW; ' % 
                   (beam_diameter*1e6, beam_divergence_fwhm * 1000, wavelength_fwhm / wavelength * 100))
    if __name__ == '__main__':
        if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
            pg.QtGui.QApplication.exec_()
elif skview:
    viewer = ImageViewer(I_display)
    viewer.show()
elif view:
    plt.imshow(I_display, interpolation='nearest', cmap='gray', origin='lower')
    plt.show()


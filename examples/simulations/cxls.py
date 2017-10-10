#!/usr/bin/env python

import sys
import time
import numpy as np
import h5py
from glob import glob

sys.path.append("../..")
import bornagain as ba
from bornagain.units import r_e, hc, keV
import bornagain.simulate.clcore as core

# Lysozyme from LCLS LO47 (Polly & Bushy) 
pdbFile = '../data/pdb/2LYZ-P1.pdb'  # Lysozyme
n_monte_carlo_iterations = 1000
n_pixels = 1500
pixel_size = 110e-6
detector_distance = 0.156
photon_energy = 9.480/keV
pulse_energy = 0.0024 
wavelength = hc/photon_energy
wavelength_fwhm = wavelength*0.001
beam_divergence_fwhm = 0.0001 # radians, full angle
beam_diameter = 1.5e-6
transmission = 0.6
n_photons = pulse_energy/photon_energy #1e12
I0 = transmission*n_photons/(np.pi*(beam_diameter/2.0)**2)
crystal_size = beam_diameter
mosaic_domain_size = 200e-9
mosaicity_fwhm = 0.0001
results_dir = '/data/temp/lcls-lysozyme/'

# John gets 2 photons per integrated peak with |F| ~ 5000

write = sys.stdout.write

# R = np.eye(3)
R = ba.utils.random_rotation()
# R = ba.utils.rotation_about_axis(0.05,[1,0,0])

# Things we probably don't want to think about
cl_group_size = 32
cl_double_precision = False

# Some overrides at the command prompt:
view = False
qtview = False
skview = False
if 'view' in sys.argv:
    view = True
if 'qtview' in sys.argv:
    import pyqtgraph as pg
    qtview = True
if 'skview' in sys.argv:
    from skimage.viewer import ImageViewer
    skview = True
if 'double' in sys.argv:
    cl_double_precision = True

# Setup simulation engine
clcore = core.ClCore(group_size=cl_group_size,double_precision=cl_double_precision)

# Setup detector geometry
panel_list = ba.detector.PanelList()
panel_list.simple_setup(n_pixels, n_pixels, pixel_size, detector_distance, wavelength)
p = panel_list[0]
q = p.Q
sa = p.solid_angle
P = p.polarization_factor

# Get atomic coordinates and scattering factors from pdb file
cryst = ba.target.crystal.structure(pdbFile)
r = cryst.r
f = ba.simulate.atoms.get_scattering_factors(cryst.Z, ba.units.hc/wavelength)

# Determine number of unit cells in whole crystal and mosaic domains
n_cells_whole_crystal = np.ceil(crystal_size/np.array([cryst.a,cryst.b,cryst.c]))
n_cells_mosaic_domain = np.ceil(mosaic_domain_size/np.array([cryst.a,cryst.b,cryst.c]))


#geom_file = results_dir + 'geom.geom'
#write('Writing geometry file %s\n' % geom_file)
#fid = open(geom_file,'w')
#fid.write("photon_energy_ev = %g\n" % photon_energy/ba.units.eV)
#fid.write("len = %g\n" % detector_distance)
#fid.write("res = %g\n" % )
#res = %g
#adu_per_ev = %g
#0/min_ss = 0
#0/max_ss = %d
#0/min_fs = 0
#0/max_fs = %d
#0/corner_x = %g
#0/corner_y = %g
#0/fs = x
#0/ss = y
#""" % (,
#       detector_distance,
#       1.0/pixel_size,
#       1.0/(photon_energy*ba.units.eV)),
#       n_pixels-1,
#       n_pixels-1,
#       -n_pixels/2.0,
#       -n_pixels/2.0)
#fid.close()

if True:
	write('Simulating molecular transform from Henke tables... ')
	t = time.time()
	q_dev = clcore.to_device(q, dtype=clcore.real_t)
	r_dev = clcore.to_device(r, dtype=clcore.real_t)
	f_dev = clcore.to_device(f, dtype=clcore.complex_t)
	a_dev = clcore.to_device(np.zeros([q_dev.shape[0]], dtype=clcore.complex_t))
	clcore.phase_factor_qrf(q_dev, r_dev, f_dev, R, a_dev)
	F2 = np.abs(a_dev.get())**2
	tf = time.time() - t
	write('%7.03f ms\n' % (tf*1e3))
else:
	write('Simulating molecular transform with cromer mann... ')
	t = time.time()
	clcore.prime_cromermann_simulator(q.copy(), cryst.Z.copy())
	q_cm = clcore.get_q_cromermann()
	r_cm = clcore.get_r_cromermann(r.copy(), sub_com=False)
	clcore.run_cromermann(q_cm, r_cm, rand_rot=False, force_rot_mat=R)
	A = clcore.release_amplitudes()
	F2 = np.abs(A)**2
	tf = time.time() - t
	write('%7.03f ms\n' % (tf*1e3))

write('Simulating lattice transform...\n')
abc = cryst.O.T.copy()
S2 = np.zeros((p.nF,p.nS))
S2_dev = clcore.to_device(S2,dtype=clcore.real_t)
message = ''
for n in np.arange(1,(n_monte_carlo_iterations+1)):

    B = ba.utils.random_beam_vector(beam_divergence_fwhm)
    w = np.random.normal(wavelength, wavelength_fwhm / 2.354820045, [1])[0]
    Rm = ba.utils.random_mosaic_rotation(mosaicity_fwhm).dot(R.copy())
    T = p.T.copy() + p.F*(np.random.random([1])-0.5) + p.S*(np.random.random([1])-0.5)

    t = time.time()
    clcore.gaussian_lattice_transform_intensities_pad(abc,n_cells_mosaic_domain, T, p.F, p.S, B, p.nF, p.nS, w, Rm, S2_dev, add=True)
    tf = time.time() - t
    sys.stdout.write('\b'*len(message))
    message = '%3.0f%% (%d; %7.03f ms)' % (n/float(n_monte_carlo_iterations)*100, n, tf*1e3)
    sys.stdout.write(message)
sys.stdout.write('\n')

# Average the lattice transforms over MC iterations
S2 = S2_dev.get().ravel()/n_monte_carlo_iterations
# Convert into useful photon units
I = I0*r_e**2*F2*S2*sa*P
# Scale up according to mosaic domain
n_domains = np.prod(n_cells_whole_crystal)/np.prod(n_cells_mosaic_domain)
I *= n_domains
I = I.reshape((p.nS, p.nF))
I = np.random.poisson(I)

if True:
    n_patterns = len(glob(results_dir+'pattern-*.h5'))
    file_name = results_dir + 'pattern-%06d.h5' % (n_patterns+1)
    write('Writing file %s\n' % file_name)
    fid = h5py.File(file_name, 'w')
#    cap = 2**14 - 2000
#    intdata = I.copy()
#    intdata[intdata > cap] = cap
#    intdata = intdata.astype(np.int32)
#    fid['/data/intdata'] = intdata
    fid['/data/data'] = I.astype(np.float32)
    fid.close()
write('Total photon counts: %g\n' % (np.sum(I)))
write('Max solid angle: %g\n' % (np.max(p.solid_angle)))
write('Min pixel intensity: %g photons\n' % (np.max(I)))
write('Max pixel intensity: %g photons\n' % (np.min(I)))
# print(np.max(I))
imdisp = np.log10(I+1e-20)
# print(np.max(imdisp))
if qtview:
    img = pg.image(imdisp,autoLevels=False,levels=[0,np.log10(500)],
              title='log10(I); %g mrad div; %g %% dE/E; ' % (beam_divergence_fwhm*1000, wavelength_fwhm/wavelength*100))
    if __name__ == '__main__':
        if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
            pg.QtGui.QApplication.exec_()
elif skview:
    viewer = ImageViewer(imdisp)
    viewer.show()
elif view:
    import matplotlib.pyplot as plt
    plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
    plt.show()

print("Done!")

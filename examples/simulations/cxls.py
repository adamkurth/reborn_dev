#!/usr/bin/env python

import sys
import time
import numpy as np

sys.path.append("../..")
import bornagain as ba
from bornagain.units import r_e, hc, keV
import bornagain.simulate.clcore as core


pdbFile = '../data/pdb/2LYZ-P1.pdb'  # Lysozyme
n_monte_carlo_iterations = 10000
n_pixels = 1000
pixel_size = 100e-6
detector_distance = 0.100
photon_energy = 12.0/keV
wavelength = hc/photon_energy
wavelength_fwhm = wavelength*0.05
beam_divergence_fwhm = 0.005 # radians, full angle
beam_diameter = 20e-6
transmission = 0.25
I0 = transmission*1e8/(np.pi*(beam_diameter/2.0)**2)
crystal_size = 20e-6
mosaic_domain_size = 1e-6
mosaicity_fwhm = 0.001

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

sys.stdout.write('Simulating molecular transform... ')
t = time.time()
q_dev = clcore.to_device(q, dtype=clcore.real_t)
r_dev = clcore.to_device(r, dtype=clcore.real_t)
f_dev = clcore.to_device(f, dtype=clcore.complex_t)
a_dev = clcore.to_device(np.zeros([q_dev.shape[0]], dtype=clcore.complex_t))
clcore.phase_factor_qrf(q_dev, r_dev, f_dev, R, a_dev)
F2 = np.abs(a_dev.get())**2
tf = time.time() - t
sys.stdout.write('%7.03f ms\n' % (tf*1e3))


sys.stdout.write('Simulating lattice transform... ')
abc = cryst.O.T.copy()
# print(abc)
S2_dev = clcore.to_device(shape=(p.nF*p.nS),dtype=clcore.real_t)
message = ''
for n in np.arange(1,(n_monte_carlo_iterations+1)):

    B = ba.utils.random_beam_vector(beam_divergence_fwhm)
    w = np.random.normal(wavelength, wavelength_fwhm / 2.354820045, [1])[0]
    Rm = ba.utils.random_mosaic_rotation(mosaicity_fwhm).dot(R.copy())
    T = p.T.copy() + p.F*(np.random.random([1])-0.5) + p.S*(np.random.random([1])-0.5)

    t = time.time()
    clcore.lattice_transform_intensities_pad(abc,n_cells_mosaic_domain, T, p.F, p.S, B, p.nF, p.nS, w, Rm, S2_dev, add=True)
    tf = time.time() - t
    sys.stdout.write('\b'*len(message))
    message = '%3.0f%% (%d; %7.03f ms)' % (n/float(n_monte_carlo_iterations)*100, n, tf*1e3)
    sys.stdout.write(message)
sys.stdout.write('\n')

# Average the lattice transforms over MC iterations
S2 = S2_dev.get()/n_monte_carlo_iterations
# Convert into useful photon units
I = I0*r_e**2*F2*S2*sa*P
# Scale up according to mosaic domain
I *= np.prod(n_cells_whole_crystal)/np.prod(n_cells_mosaic_domain)


I = I.reshape((p.nS, p.nF))
# imdisp = S2.reshape((p.nS, p.nF))
I = np.random.poisson(I)
# I[I > (2**16-1)] = 2**16-1
# print(np.max(I))
imdisp = np.log10(I+1)
if qtview:
    img = pg.image(imdisp,autoLevels=False,levels=[0,2],
             title='log10(I+1); %g mrad div; %g %% dE/E; ' % (beam_divergence_fwhm*1000, wavelength_fwhm/wavelength*100))
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
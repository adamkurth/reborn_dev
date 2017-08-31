import sys
import time

import numpy as np
import afnumpy as afnp
import matplotlib.pyplot as plt
# import pyqtgraph as pg
import pyopencl as cl
# import pyopencl.clmath as clmath

sys.path.append("../..")
import bornagain as ba
import bornagain.target.crystal as crystal

try:
    import bornagain.simulate.clcore as core
except:
    print('Cannot import clcore; check that pyopencl is installed')

print(cl)

show = True   # Display the simulated patterns
double = False # Use double precision if available
if 'view' in sys.argv: show = True
if 'double' in sys.argv: double = True

clcore = core.ClCore(group_size=32,double_precision=double)

# Settings for the pixel-array detector
n_pixels = 1000
pixel_size = 100e-6
detector_distance = 0.05
wavelength = 1.5e-10
panel_list = ba.detector.PanelList()
panel_list.simple_setup(n_pixels, n_pixels, pixel_size, detector_distance, wavelength)
p = panel_list[0]

# Load a crystal structure from pdb file
pdb_file = '../data/pdb/2LYZ.pdb'  # Lysozyme
cryst = crystal.structure(pdb_file)

# These are atomic coordinates (Nx3 array)
r = cryst.r

# Look up atomic scattering factors from the Henke tables (they are complex numbers)
f = ba.simulate.atoms.get_scattering_factors(cryst.Z,ba.units.hc/panel_list.beam.wavelength)

n_trials = 3
show_all = show
plt.ion()

# imwin = pg.image()

n_pixels = p.nF*p.nS
n_atoms = r.shape[0]
r = clcore.to_device(r, dtype=clcore.real_t)
f = clcore.to_device(f, dtype=clcore.complex_t)
A = clcore.to_device(shape=[n_pixels],dtype=clcore.complex_t)
I = clcore.to_device(shape=[n_pixels],dtype=clcore.real_t)
# Isum = clcore.to_device(shape=[n_pixels],dtype=clcore.real_t)

N = 3
for n in np.arange(1,(N+1)):

    R = None #ba.utils.random_rotation_matrix()
    t = time.time()
    clcore.phase_factor_pad(r, f, p.T, p.F, p.S, p.B, p.nF, p.nS, panel_list.beam.wavelength, R, A)
    tf = time.time() - t
    print('%d phase_factor_pad: %7.03f ms (%d atoms; %d pixels)' % (n, tf*1e3,n_atoms,n_pixels))
#     clcore.mod_squared_complex_to_real(A,I)
#     Isum += I

#     # Display pattern
#     if n % 10 == 0:
#         imdisp = I.get()/n
#         imdisp = imdisp.reshape((p.nS, p.nF))
#         imdisp = np.log(imdisp + 0.1)
#         plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
#         plt.title('y: up, x: right, z: beam (towards you)')
#         plt.show()
#         print('plot')
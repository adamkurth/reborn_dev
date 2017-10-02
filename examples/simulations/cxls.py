import sys
import time
import numpy as np

sys.path.append("../..")
import bornagain as ba
import bornagain.target.crystal as crystal
import bornagain.simulate.clcore as core
import matplotlib.pyplot as plt
import pyqtgraph as pg

do_monte_carlo = True
n_monte_carlo_iterations = 1000
n_cells = 20
show = True   # Display the simulated patterns
double = False # Use double precision if available
if 'view' in sys.argv: show = True
if 'double' in sys.argv: double = True
cl_group_size = 32
cl_double_precision = False
n_pixels = 1500
pixel_size = 5e-6
detector_distance = 0.05
wavelength = 2.0e-10
wavelength_fwhm = wavelength*0#.05
beam_fwhm = 0.001 # radians

print("Setup opencl programs")
clcore = core.ClCore(group_size=cl_group_size,double_precision=cl_double_precision)
crystal_size = np.array([n_cells,n_cells,n_cells],dtype=clcore.int_t)
print('crystal size',crystal_size)

print('Setup detector panel list...')
panel_list = ba.detector.PanelList()
panel_list.simple_setup(n_pixels, n_pixels, pixel_size, detector_distance, wavelength)
p = panel_list[0]
print(panel_list)

d = 200e-10
abc = np.array([[d,0,0],[0,d,0],[0,0,d]],dtype=clcore.real_t)

n_pixels = p.nF*p.nS
I = clcore.to_device(shape=[n_pixels],dtype=clcore.real_t)

B0 = np.array([0,0,1])
def random_B(div_fwhm):
    if div_fwhm == 0:
        return(np.array([0,0,1.0]))
    sig = div_fwhm/2.354820045
    a = np.random.normal(0,sig,[2])
    B = np.array([a[0],a[1],1])
    B /= np.sqrt(np.sum(B**2))
    return B

wavelength = panel_list.beam.wavelength
def random_wavelength(w0,fwhm):
    if fwhm == 0:
        return w0
    return np.random.normal(w0,fwhm/2.354820045,[1])

R = np.eye(3,dtype=clcore.real_t)

if not do_monte_carlo:
    beam_fwhm = 0
    wavelength_fwhm = 0

for n in np.arange(1,(n_monte_carlo_iterations+1)):

    B = random_B(beam_fwhm)
    w = random_wavelength(wavelength,wavelength_fwhm)
    t = time.time()
    clcore.lattice_transform_intensities_pad(abc,crystal_size, p.T, p.F, p.S, B, p.nF, p.nS, w, R, I, add=True)
    tf = time.time() - t
    print('%d: %7.03f ms' % (n, tf*1e3))

I = I.get()/n
I = I.reshape((p.nS, p.nF))
imdisp = np.log(I + 1e5)
imdisp[np.isinf(imdisp)] = n_cells^3
pg.image(imdisp)
# plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
# plt.show()


print("done")
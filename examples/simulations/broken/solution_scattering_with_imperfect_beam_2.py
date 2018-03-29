import sys
import time
import numpy as np

sys.path.append("../..")
import bornagain as ba
import bornagain.target.crystal as crystal
import bornagain.simulate.clcore as core
import matplotlib.pyplot as plt

N = 1000
show = True   # Display the simulated patterns
double = False # Use double precision if available
if 'view' in sys.argv: show = True
if 'double' in sys.argv: double = True

clcore = core.ClCore(group_size=32,double_precision=double)

# Settings for the pixel-array detector
n_pixels = 500
pixel_size = 200e-6
detector_distance = 0.1
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

n_pixels = p.nF*p.nS
n_atoms = r.shape[0]
r = clcore.to_device(r, dtype=clcore.real_t)
f = clcore.to_device(f, dtype=clcore.complex_t)
A = clcore.to_device(shape=[n_pixels],dtype=clcore.complex_t)
I = clcore.to_device(shape=[n_pixels],dtype=clcore.real_t)
Isum = clcore.to_device(shape=[n_pixels],dtype=clcore.real_t)

profile = ba.scatter.RadialProfile()
profile.make_plan(panel_list,nBins=300)

B0 = np.array([0,0,1])
beam_fwhm = 0.01 # radians
def random_B(div_fwhm):
    if div_fwhm == 0:
        return(np.array([0,0,1.0]))
    sig = div_fwhm/2.354820045
    a = np.random.normal(0,sig,[2])
    B = np.array([a[0],a[1],1])
    B /= np.sqrt(np.sum(B**2))
    return B

wavelength = panel_list.beam.wavelength
wavelength_fwhm = wavelength*0.01
def random_wavelength(w0,fwhm):
    if fwhm == 0:
        return w0
    return np.random.normal(w0,fwhm/2.354820045,[1])

for n in np.arange(1,(N+1)):

    R = ba.utils.random_rotation_matrix()
    t = time.time()
    clcore.phase_factor_pad(r, f, p.T, p.F, p.S, B0, p.nF, p.nS, wavelength, R, A)
    tf = time.time() - t
    print('%d phase_factor_pad: %7.03f ms (%d atoms; %d pixels)' % (n, tf*1e3,n_atoms,n_pixels))
    clcore.mod_squared_complex_to_real(A,I)
    Isum += I

I0 = I.get()/n
prof = profile.get_profile(I0)
lprof = np.log(prof)
plt.plot(lprof/np.sum(lprof))

ws = np.zeros([N])

for n in np.arange(1,(N+1)):

    R = ba.utils.random_rotation_matrix()
    B = random_B(beam_fwhm)
    w = random_wavelength(wavelength,wavelength_fwhm)
    ws[n-1] = w
    t = time.time()
    clcore.phase_factor_pad(r, f, p.T, p.F, p.S, B, p.nF, p.nS, w, R, A)
    tf = time.time() - t
    print('%d phase_factor_pad: %7.03f ms (%d atoms; %d pixels)' % (n, tf*1e3,n_atoms,n_pixels))
    clcore.mod_squared_complex_to_real(A,I)
    Isum += I


I = I.get()/n
prof = profile.get_profile(I)
lprof = np.log(prof)
plt.plot(lprof/np.sum(lprof))
plt.show()

# plt.plot(ws,'.')
# plt.show()

# I = I.reshape((p.nS, p.nF))
# imdisp = np.log(I + 100)
# plt.imshow(imdisp, interpolation='nearest', cmap='gray', origin='lower')
# plt.title('y: up, x: right, z: beam (towards you)')
# plt.show()


print("done")
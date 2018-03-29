import sys
import time

import numpy as np
from numpy.random import randn, random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

sys.path.append("../..")
import bornagain as ba
from bornagain.utils import vec_norm, random_rotation_matrix
from bornagain.simulate.clcore import ClCore

clcore = ClCore()

# Create a detector
panel = ba.detector.Panel()
n_pixels = 1000;
pixel_size = 100e-6
detector_distance = 0.1
wavelength = 1.5e-10
wavelength_sigma = wavelength/500.0  
divergence = 0.001
panel.nF = n_pixels
panel.nS = n_pixels
panel.F = np.array([1,0,0])*pixel_size
panel.S = np.array([0,1,0])*pixel_size
panel.T = [0,0,detector_distance]
panel.beam.B = [0,0,1]
panel.beam.wavelength = wavelength

# Scattering vectors

def mcQ(panel,wavelength,wavelength_sigma,divergence,solid_angle=True):
    """ Monte Carlo q vectors; add jitter to wavelength, pixel position,
    incident beam direction for each pixel independently. """
 
    i = np.arange(panel.nF,dtype=np.float)
    j = np.arange(panel.nS,dtype=np.float)
    [i, j] = np.meshgrid(i,j)
    i = i.ravel()
    j = j.ravel()
    # this jitters the pixel position to emulate solid angle
    if solid_angle:
        i += random(panel.n_pixels) - 0.5
        j += random(panel.n_pixels) - 0.5
    
    F = np.outer(i, panel.F)
    S = np.outer(j, panel.S)
    T = panel.T
    V = T + F + S
    # This is a horrible way to jitter the incident beam direction...
    B = np.outer(np.ones(panel.n_pixels), panel.B) + \
        randn(panel.n_pixels, 3) * divergence
    K = vec_norm(V) - vec_norm(B)
    # this varies the wavelength
    lam = wavelength + randn(panel.n_pixels) * wavelength_sigma
    
    return 2 * np.pi * K / np.outer(lam, np.ones(3))

# Atomic coordinates
N = 10
x = np.arange(0,N)*30e-10
[xx,yy,zz] = np.meshgrid(x,x,x,indexing='ij')
r = np.zeros([N**3,3])
r[:,0] = zz.flatten()
r[:,1] = yy.flatten()
r[:,2] = xx.flatten()

# Scattering factors
f = np.ones([N**3], dtype=np.complex64)

# Rotation matrix which acts on q vectors
R = random_rotation_matrix() #np.eye(3)

# First a simulation without monte carlo
q = panel.Q
A = clcore.phase_factor_qrf(q,r,f,R)
I0 = np.abs(A)**2


mc_iterations = 3
I = 0
for iter in range(mc_iterations):
    # Compute diffraction amplitudes
    # This can be ** much ** faster if we don't move data between cpu and gpu..
    t = time.time()
    q = mcQ(panel,wavelength,wavelength_sigma,divergence,True)
    A = clcore.phase_factor_qrf(q,r,f,R)
    I += np.abs(A)**2
    print(time.time() - t)


# Multiply by polarization factor
I *= panel.polarization_factor




dispim = np.reshape(I0,[n_pixels,n_pixels])
dispim = np.log10(dispim+10)
plt.imshow(dispim,interpolation='nearest',cmap='gray',origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()

dispim = np.reshape(I,[n_pixels,n_pixels])
dispim = np.log10(dispim+10)
plt.imshow(dispim,interpolation='nearest',cmap='gray',origin='lower')
plt.title('y: up, x: right, z: beam (towards you)')
plt.show()
"""
Script to do basic phasing - can be used as a template for further
phasing explorations.
2D as of 2018

Date Created: 14 Apr 2018
Last Modified: 28 Mar 2018
Humans Responsible: Joe P. Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift
from skimage import data, transform

# Phase retrieval algorithm parameters
n_iterations = 200  # What is this?
cycle = 50
it_IPA = 45
# Difference map update parameters
beta_DM = 0.95
gamma_M = -1 / beta_DM
gamma_S = 1 / beta_DM

# =====================================================================
# Create the object that we will try to image with our IPA
# =====================================================================
# Image size
nx = 64
ny = 64
oversampling = 4
# Over-sampled image size
nx_os = nx * oversampling
ny_os = ny * oversampling
# Fetch an image of cameraman
img = data.camera()
img = transform.resize(img, (nx, ny))
img = img / np.max(img)
# This is the true solution:
x_true = np.zeros([nx_os, ny_os])  # Oversampled
x_true[0:nx, 0:ny] = img

# =====================================================================
# Inputs to the phase retrieval algorithm
# =====================================================================
# Support (region where the object density is non-zero)
x_supp = np.zeros([nx_os, ny_os])
x_supp[0:nx, 0:ny] = np.ones([nx, ny])
# Calculate diffraction magnitude data (square root of intensities)
M_data = np.abs(fftn(x_true))

# =====================================================================
# Define the projection operators
# =====================================================================
def P_S(x, x_supp):
    # Support projection operator.  It simply sets all values to zero
    # if they are outside of the support.
    x_new = x * x_supp
    return x_new

def P_M(x, M_data):
    # Fourier modulus projection operator.  It corrects the fourier
    # transform so that the modulus is equal to the measured one.
    X = fftn(x)
    M = np.abs(X)
    X_new = (X / M) * M_data
    x_new = ifftn(X_new)
    return x_new

def R_M(x, gamma_M, M_data):
    # Relaxed modulus projection operator.
    return (1 + gamma_M) * P_M(x, M_data) - gamma_M * x

def R_S(x, gamma_S, x_supp):
    # Relaxed support projection operator.
    return (1 + gamma_S) * P_S(x, x_supp) - gamma_S * x

# Initialize error storage and error comparisons
errors = np.zeros(n_iterations)
error_best = np.inf
# Initialize iterate (random numbers)
x = np.random.rand(nx_os, ny_os)
x0 = x.copy()  # Keep a copy of the starting point
x_best = x0.copy()  # Best reconstruction (so far)
# Calculate initial error
M_est = fftn(np.abs(x))
errors[0] = np.sum((np.abs(x[0:nx, 0:ny] - x_true[0:nx, 0:ny])) ** 2)
# Start of IPA loop
for it in range(1, n_iterations):
    # Projection algorithm switching
    if np.mod(it, cycle) < it_IPA:  # Difference Map algorithm
        IPA_curr = 'DM'
        x_PM = P_M(R_S(x, gamma_S, x_supp), M_data)
        x_PS = P_S(R_M(x, gamma_M, M_data), x_supp)
        x = x + beta_DM * (x_PM - x_PS)
    else:  # Error reduction algorithm
        IPA_curr = 'ER'
        x_PM = P_M(x, M_data)
        x_PS = P_S(x_PM, x_supp)
        x = x_PS
    # Error calculation
    errors[it] = np.sum((np.abs(x[0:nx, 0:ny] - x_true[0:nx, 0:ny])) ** 2)
    # Keep the best reconstruction so far as measured by the data error
    if errors[it] < error_best:
        x_best = x.copy()
        error_best = errors[it]
    print("iteration = %d, %s, error = %.2e" % (it, IPA_curr, errors[it]))

# Normalise errors
errors = np.sqrt(errors / np.sum(x_true[0:nx, 0:ny] ** 2))

# =====================================================================
# Visualise results
# =====================================================================
# Plot the errors
fig, ax = plt.subplots(1, 1)
ax.plot(np.arange(0, n_iterations), np.log10(errors), lw=2, color='b', linestyle='-', label='e')
plt.xlabel('Iteration')
plt.ylabel(r'$\log_{10}$(Error)')
ax.grid()
# Show the true object and diffraction magnitudes
fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(x_true[0:nx, 0:ny], interpolation='nearest')
ax.set_title('Object')
ax = fig.add_subplot(222)
ax.imshow(np.log10(fftshift(M_data)), interpolation='nearest')
ax.set_title('Diffraction')
ax = fig.add_subplot(223)
ax.imshow(np.abs(x0[0:nx, 0:ny]), interpolation='nearest')
ax.set_title("Initial guess")
ax = fig.add_subplot(224)
ax.imshow(np.abs(x_best[0:nx, 0:ny]), interpolation='nearest')
ax.set_title("Reconstruction")
plt.show()

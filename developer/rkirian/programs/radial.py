# Radial equation to 2-D with Poisson noise

import numpy as np
import matplotlib.pyplot as plt

def sinc(r):
    if r==0:
        return 101
    else:
        return  (100/r) * np.sin(r/50) + 5

def cos(x):
    return 10*np.cos(np.pi*x/100) + 11

def two_d(f, x, y):
    z = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            z[i,j] = f(np.sqrt(((x/2)-i)**2 + ((y/2)-j)**2))
    return z

def add_poisson_noise(z):
    zsum = np.sum(z)
    zplusnoise = np.random.poisson(lam = z, size=None)
    zplussum = np.sum(zplusnoise)
    zplusnoise = zplusnoise * (zsum/zplusnoise)
    if zsum == np.sum(zplusnoise):
        print("True")
    else:
        print("False")
    return zplusnoise

def add_gaussian_noise(z, sigma):
    z = np.random.normal(z, sigma)
    return z

def graph_image(z):
    image = plt.imshow(z)
    image.set_cmap('nipy_spectral')
    plt.colorbar()
    plt.show()

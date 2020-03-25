import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from reborn.viewers.qtviews import MapProjection
import matplotlib.pyplot as plt
import pyqtgraph as pg

filename = 'run0001_intensity.npz'
dat1 = np.load(filename)
print(dat1.files)
filename = 'run0001_au_amplitude.npz'
dat2 = np.load(filename)
amps = dat2['map']
dens = ifftn(ifftshift(amps))
MapProjection(np.abs(dens))
# plt.scatter(np.arange(len(dens.ravel())), np.abs(dens).ravel(), label='abs')
MapProjection(np.real(dens))
pg.plot(np.arange(len(dens.ravel())), np.real(dens).ravel())
pg.mkQApp().exec_()
MapProjection(np.imag(dens))
# plt.scatter(np.arange(len(dens.ravel())), np.imag(dens).ravel(), label='imag')
# plt.legend()

# print(np.max((np.abs(dens)-np.real(dens))/np.real(dens)))
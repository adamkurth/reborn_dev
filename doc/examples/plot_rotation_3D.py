r"""
3D rotation of a density map
===============================

Simple diffraction simulation from lattice of point scatterers.

Contributed by Joe Chen and Kevin Schmidt

Imports:
"""

import reborn as ba

import numpy as np
from reborn.target import crystal
import scipy.constants as const

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from numpy.fft import fftn, ifftn, fftshift, ifftshift, fft, ifft



plt.close('all')


eV = const.value('electron volt')

pdb_file = '1jb0.pdb'
resolution = 3e-10
oversampling = 1
photon_energy_ev = 12000


cryst = crystal.CrystalStructure(pdb_file, tight_packing=True)
uc = cryst.unitcell
sg = cryst.spacegroup
print(uc)
print(sg)


cdmap = crystal.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=oversampling)


f = cryst.molecule.get_scattering_factors(photon_energy=photon_energy_ev*eV)
print('sum of f', np.sum(f))

rho_au = cdmap.place_atoms_in_map(cryst.fractional_coordinates, f, mode='trilinear')
print('sum of rho_au', np.sum(rho_au))




#==================

CMAP = "viridis"

print(rho_au.shape)
Nx, Ny, Nz = rho_au.shape
Nx_cent = int(np.round(Nx/2))
Ny_cent = int(np.round(Ny/2))
Nz_cent = int(np.round(Nz/2))

def show_slice(disp_map, disp_str):
    """
    Slice
    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    im = ax.imshow(disp_map[Nx_cent,:,:], interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[Nx_cent,:,:]')
    ax = fig.add_subplot(132)
    im = ax.imshow(disp_map[:,Ny_cent,:], interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,Ny_cent,:]')
    ax = fig.add_subplot(133)
    im = ax.imshow(disp_map[:,:,Nz_cent], interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,:,Nz_cent]')

    plt.suptitle(disp_str)
    plt.tight_layout()
    plt.show()

def show_projection(disp_map, disp_str):
    """
    Projection
    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    im = ax.imshow(np.sum(disp_map,axis=0), interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[Nx_cent,:,:]')
    ax = fig.add_subplot(132)
    im = ax.imshow(np.sum(disp_map,axis=1), interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,Ny_cent,:]')
    ax = fig.add_subplot(133)
    im = ax.imshow(np.sum(disp_map,axis=2), interpolation='nearest', cmap=CMAP, origin='lower')
    fig.colorbar(im, shrink=0.5, ax=ax)
    ax.set_title('[:,:,Nz_cent]')

    plt.suptitle(disp_str)
    plt.tight_layout()
    plt.show()


disp_map = np.abs(rho_au)
disp_str = 'Asymmetric unit: projection'
show_projection(disp_map, disp_str)




Nx, Ny, Nz = rho_au.shape

N = Nx


f = np.zeros((N,N,N))
f[0:Nx, 0:Ny, 0:Nz] = rho_au


Nx_os = N
Ny_os = N
Nz_os = N

Ny_sta = 0
Ny_end = N

Nz_sta = 0
Nz_end = N

# yay
#==================
# Rotation



def rotate90(f):
   return np.transpose(np.fliplr(f))

def rotate180(f):
   return np.fliplr(np.flipud(f))

def rotate270(f):
   return np.transpose(np.flipud(f))


# Precalculations
Y, X = np.meshgrid(np.arange(N), np.arange(N))

y0 = 0.5*(N-1)
constx1 = -1j*2.0*np.pi/N * X * (Y-y0)
constx2_1 = -1j*np.pi * (1-(N%2)/N)
constx2_2 = constx2_1*X
constx2_3 = constx2_1*(Y-y0)

x0 = 0.5*(N-1)
consty1 = -1j*2.0*np.pi/N * Y * (X-x0)
consty2_1 = -1j*np.pi * (1-(N%2)/N)
consty2_2 = consty2_1*Y
consty2_3 = consty2_1*(X-x0)

TwoOverPi = 2.0/np.pi
PiOvTwo = np.pi*0.5


def shiftx(f,kxfac,xfac):
	return ifft(fftshift(fft(f, axis=0), axes=0) * kxfac, axis=0) * xfac


def shifty(f,kyfac,yfac):
	return ifft(fftshift(fft(f, axis=1), axes=1) * kyfac, axis=1) * yfac


def rotate2D(fr, kxfac, xfac, kyfac, yfac, n90_mod_Four):
	
	if (n90_mod_Four == 1):
		fr = rotate90(fr)
	elif (n90_mod_Four == 2):
		fr = rotate180(fr)
	elif (n90_mod_Four == 3):
		fr = rotate270(fr)

	fr = shiftx(fr, kxfac, xfac)
	fr = shifty(fr, kyfac, yfac)
	fr = shiftx(fr, kxfac, xfac)

	return fr


def rotate_Euler_z(f, ang):

	n90 = np.rint(ang*TwoOverPi)
	dang = ang - n90 * PiOvTwo

	t = -np.tan(0.5*dang)
	s = np.sin(dang)

	kxfac = np.exp(constx1 * t)
	xfac = np.exp(constx2_2 - constx2_3 * t)

	kyfac = np.exp(consty1 * s)
	yfac = np.exp(consty2_2 - consty2_3 * s)

	n90_mod_Four = n90 % 4

	f_rot = np.zeros((Nx_os, Ny_os, Nz_os), dtype=np.complex128)
	for ii in range(Nz_sta, Nz_end):
		f_rot[ii,:,:] = rotate2D(f[ii,:,:], kxfac, xfac, kyfac, yfac, n90_mod_Four)

	return f_rot


def rotate_Euler_y(f, ang):

	n90 = np.rint(ang*TwoOverPi)
	dang = ang - n90 * PiOvTwo

	t = -np.tan(0.5*dang)
	s = np.sin(dang)

	kxfac = np.exp(constx1 * t)
	xfac = np.exp(constx2_2 - constx2_3 * t)

	kyfac = np.exp(consty1 * s)
	yfac = np.exp(consty2_2 - consty2_3 * s)

	n90_mod_Four = n90 % 4

	f_rot = np.zeros((Nx_os, Ny_os, Nz_os), dtype=np.complex128)
	for ii in range(Ny_sta, Ny_end):
		f_rot[:,ii,:] = rotate2D(f[:,ii,:], kxfac, xfac, kyfac, yfac, n90_mod_Four)

	return f_rot


def rotate3D(f, EulerAng_vec):

	f_rot = rotate_Euler_z(f, ang=EulerAng_vec[0])
	f_rot = rotate_Euler_y(f_rot, ang=EulerAng_vec[1])
	f_rot = rotate_Euler_z(f_rot, ang=EulerAng_vec[2])

	return f_rot







# theta_rot = 90 *(np.pi/180)
# x_rot = rotate3D(f=f, 
# 				 EulerAng_vec=np.array([theta_rot,0,0]))


x_rot = rotate3D(f=f, 
				 EulerAng_vec=np.array([20,0,0])*(np.pi/180))


disp_map = np.abs(x_rot)
disp_str = 'Asymmetric unit: projection'
show_projection(disp_map, disp_str)

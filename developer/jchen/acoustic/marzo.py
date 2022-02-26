"""
Simulating acoustic pressure according to Marzo

Date Created: 25 Feb 2022
Last Modified: 26 Feb 2022
Author: Joe Chen
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Plotting parameters
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
CMAP = 'viridis'
font_size = 12
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
CMAP = "viridis"


def colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

def show_slice(disp_map, disp_str, slice_ind, clim=None):
    """
    Plot the three orthogonal slices
    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    im = ax.imshow(disp_map[slice_ind[0],:,:], clim=clim, cmap=CMAP, origin='lower', extent=[z_min*1e3,z_max*1e3,x_min*1e3,x_max*1e3])
    colorbar(ax, im)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('x (mm)')
    # ax.set_title(f'[{slice_ind[0]},:,:]')
    ax = fig.add_subplot(132)
    im = ax.imshow(disp_map[:,slice_ind[1],:], clim=clim, cmap=CMAP, origin='lower', extent=[z_min*1e3,z_max*1e3,y_min*1e3,y_max*1e3])
    colorbar(ax, im)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('y (mm)')
    # ax.set_title(f'[:,{slice_ind[1]},:]')
    ax = fig.add_subplot(133)
    im = ax.imshow(disp_map[:,:,slice_ind[2]], clim=clim, cmap=CMAP, origin='lower', extent=[x_min*1e3,x_max*1e3,y_min*1e3,y_max*1e3])
    colorbar(ax, im)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    # ax.set_title(f'[:,:,{slice_ind[2]}]')

    plt.suptitle(disp_str)
    plt.tight_layout()
    plt.show(block=False)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def pressure_one_piston(piston_pos, piston_normal, phase, P0, Vpp, k_wv):
    """ Calculate the pressure field for one piston """

    # Calculating theta
    Theta = np.zeros((Nx,Ny,Nz), dtype=np.double)
    for i in range(Nx):
        print(i/Nx)
        for j in range(Ny):
            for k in range(Nz):
                r = np.array([X[i,j,k],Y[i,j,k],Z[i,j,k]]) - piston_pos
                Theta[i,j,k] = angle_between(v1=r, v2=piston_normal)

    # Calculating d
    d = np.sqrt((X-piston_pos[0])**2+(Y-piston_pos[1])**2+(Z-piston_pos[2])**2)

    Df = np.sinc(k_wv*a*np.sin(Theta) / np.pi) #numpy sinc is defined as sin(pi x)/ (pi x)

    return P0 * Vpp * (Df/d) * np.exp(1j*(phase + k_wv*d))



# Df = 2 * J1(k*a*np.sin(Theta)) / (k * a * np.sin(Theta))



#=======================================================================

P0 = 1    # Amplitude constant that defines transducer power output
Vpp = 1   # Excitation signal peak-to-peak amplitude

a = 4.5e-3     # radius of the piston 
lamb = 8.65e-3 # wavelength of the ultrasound in m

k_wv = 2*np.pi/lamb

x_max = 20e-3
y_max = 20e-3
z_max = 20e-3

x_min = -20e-3
y_min = -20e-3
z_min = -20e-3

x_step = 0.5e-3
y_step = 0.5e-3
z_step = 0.5e-3

x = np.arange(x_min,x_max,x_step)
y = np.arange(y_min,y_max,y_step)
z = np.arange(z_min,z_max,z_step)
X,Y,Z = np.meshgrid(x,y,z)

Nx, Ny, Nz = X.shape
Nx_cent = int(Nx/2)
Ny_cent = int(Ny/2)
Nz_cent = int(Nz/2)



piston_normal = np.array([0,0,-1])
phase = 0
piston_pos = np.array([0,0,2])*1e-2
P1 = pressure_one_piston(piston_pos, piston_normal, phase, P0, Vpp, k_wv)


piston_normal = np.array([0,0,1])
phase = -(2/13)*np.pi
piston_pos = np.array([0,0,-2])*1e-2
P2 = pressure_one_piston(piston_pos, piston_normal, phase, P0, Vpp, k_wv)


P_sum = P1+P2

show_slice(disp_map=np.abs(P_sum), disp_str='sum|P(r)|', slice_ind=[Nx_cent,Ny_cent,Nz_cent], clim=None)
show_slice(disp_map=np.log10(np.abs(P_sum)), disp_str='log sum|P(r)|', slice_ind=[Nx_cent,Ny_cent,Nz_cent], clim=[1,3])

# show_slice(disp_map=np.log10(np.abs(P_sum)), disp_str='log sum|P(r)|', slice_ind=[Nx_cent,Ny_cent,Nz_cent], clim=[1,5])













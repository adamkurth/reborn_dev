"""
Testing 3D rotation with the shears implemented using linear interpolation rather than FFTs.

Date Created: 23 Oct 2021
Last Modified: 23 Oct 2021
Humans responsible: JC
"""

import numpy as np

from reborn.misc.rotate import Rotate3D, Rotate3DLinearShear
import matplotlib.pyplot as plt
import scipy

plt.close('all')

#-----------------------
# Set up - a 3D Guassian
x = np.arange(-5,5,0.3)
y = np.arange(-5,5,0.3)
z = np.arange(-5,5,0.3)

X,Y,Z = np.meshgrid(x,y,z)

shift_vec = [0, -2.36, 0.97]
sigma_vec = [0.1, 0.35, 0.21]

rho = np.exp(-(X-shift_vec[0])**2/(2*sigma_vec[0]) 
	         -(Y-shift_vec[1])**2/(2*sigma_vec[1]) 
	         -(Z-shift_vec[2])**2/(2*sigma_vec[2]))


Nx, Ny, Nz = rho.shape
Nx_cent = int(Nx/2)

plt.figure()
# plt.imshow(rho[Nx_cent, :, :])
# plt.imshow(rho[:, Nx_cent, :])
plt.imshow(rho[:, :, Nx_cent])
plt.title('Original Gaussian')
plt.show(block=False)

#-----------------------
# Rotation parameters

phi = 30*np.pi/180.0
c = np.cos(phi)
s = np.sin(phi)

# R = np.array([[c, 0, s],
#               [0, 1, 0],
#               [-s, 0, c]])

R = np.array([[c, -s, 0],
              [s, c, 0],
              [0, 0, 1]])
Rs = scipy.spatial.transform.Rotation.from_matrix(R)


rot_obj = Rotate3D(rho)
rot_obj.rotation(Rs)
rho_rotated = (np.abs(rot_obj.f))

plt.figure()
# plt.imshow(rho_rotated[Nx_cent, :, :])
# plt.imshow(rho_rotated[:, Nx_cent, :])
plt.imshow(rho_rotated[:, :,Nx_cent])
plt.title('Rotated by FFT shears')
plt.show(block=False)


#-----------------------
# Now try the linear interpolated shear

print('Linear interp shear')
rot_ls_obj = Rotate3DLinearShear(rho)
rot_ls_obj.rotation(Rs)
rho_ls_rotated = (np.abs(rot_ls_obj.f))

plt.figure()
# plt.imshow(rho_ls_rotated[Nx_cent, :, :])
# plt.imshow(rho_ls_rotated[:, Nx_cent, :])
plt.imshow(rho_ls_rotated[:, :,Nx_cent])
plt.title('Rotated by linear interp shears')
plt.show(block=False)


print(np.sum(np.abs(rho_ls_rotated - rho_rotated)))







# Shear test
"""
A = x**2
A_new = np.zeros_like(A)

N = len(A)
theta = 170/180 * np.pi

yy = 1
for xx in range(N):
	# for yy in range(1):
	xx_new = xx - np.tan(theta/2) * yy
	xx_new_int = int(xx_new)
	Delta = xx_new - xx_new_int
	A_new[xx] = A[xx_new_int] * (1-Delta) + A[xx_new_int+1] * Delta


plt.plot(A)
plt.plot(A_new)
plt.show()
"""



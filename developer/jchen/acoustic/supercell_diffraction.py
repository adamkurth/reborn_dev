import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift

plt.close('all')

# Units in SI

# Todo: make samples not depend on num of particles

#================================
n_part = 40 # number of particles
n_samp = 100

a = 0.2e-6 # particle spacing
r_max = 8e-6 #0.5e-3 # X-ray beam spatial coherence width 
lamb = 1e-6
r = np.linspace(0, r_max, n_samp)

#================================

def position_molecule(r, lamb, A=1):
	# r_part = r.copy() # The original position is zero (origin)

	pressure = A*np.cos(2*np.pi/lamb * r)

	pressure_gradient = -A*2*np.pi/lamb * np.sin(2*np.pi/lamb * r)
	scale = 0.5*lamb*(lamb/(2*np.pi)) #5e-5

	r_part = np.zeros(n_part)
	for i in range(n_part):
		r_part[i] = a * i
		r_part[i] += scale*pressure_gradient[i]

		# print(r_part[i])
		# if i == 2:
		# 	yay

	return r_part, pressure


"""
r_part = r.copy() # The original position is zero (origin)

A = 1
pressure = A*np.sin(2*np.pi/lamb * r)
pressure_gradient = A*2*np.pi/lamb * np.cos(2*np.pi/lamb * r)
d_min = 0#0.1
scale = (lamb/(A*2*np.pi))**2#5e-5

for i in range(n_part):
	r_part[i] = r_part[i] + scale*pressure_gradient[i] + d_min
"""


r_part0, pressure0 = position_molecule(r, lamb=1e-8, A=0)
r_part, pressure = position_molecule(r, lamb=lamb, A=1)
# r_part2 = r.copy()


# plt.figure()
# plt.plot(r_part, np.zeros(n_part), 'o')
# plt.plot(r, pressure, '--', color='tab:orange')
# plt.grid()
# plt.show(block=False)


# plt.figure()
# plt.plot(r, np.zeros(n_part), 'o')
# # plt.plot(r, pressure, '--', color='tab:orange')
# plt.grid()
# plt.show(block=False)


#================================
# Make the electron density
# f[i] = np.ones(n_part) # Electron density - each particle are the same

f = np.zeros(n_part)
for i in range(n_part):
	i_mod = i % 4
	if i_mod == 0:
		f[i] = 1
	elif i_mod == 1:
		f[i] = 1
	elif i_mod == 2:
		f[i] = 1
	elif i_mod == 3:
		f[i] = 1


#================================
# Calculate the diffraction

q_max = 2*np.pi/lamb * 10
q = np.linspace(0, q_max, n_samp)


def calc_FT(r, f, q):
	F = np.zeros(n_samp, dtype=np.complex128)

	for j in range(n_samp):
		print(j/n_samp)
		for i in range(n_part):
			F[j] += f[i] * np.exp(1j * q[j] * r[i])
	return F

F = calc_FT(r=r_part, f=f, q=q)
F0 = calc_FT(r=r_part0, f=f, q=q)

# plt.figure()
# plt.plot(q, np.abs(F), 'o-')
# plt.grid()
# plt.show(block=False)







fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(r_part, np.zeros(n_part), 'o')
# axs[0, 0].plot(r_part, f, '-')
axs[0, 0].plot(r, pressure, '--', color='tab:orange')
axs[0, 0].grid()

axs[1, 0].plot(q, np.abs(F)**2, '-')
axs[1, 0].grid()
# axs[1, 0].set_title("shares x with main")


axs[0, 1].plot(r_part0, np.zeros(n_part), 'o')
# axs[0, 1].plot(r_part2, f, '-')
# axs[0, 1].plot(r, pressure2, '--', color='tab:orange')
axs[0, 1].grid()

axs[1, 1].plot(q, np.abs(F0)**2, '-')
axs[1, 1].grid()
# axs[1, 0].set_title("shares x with main")

fig.tight_layout()
plt.show(block=False)

# yay

"""
num_uc = 10
f = np.array([1,5,3,2])
c = np.tile(f,num_uc)
num_samples = len(c)*10

r_max = 10 # some unit
r = np.linspace(0,r_max,num_samples)
density = np.sin(r)

d_min = 0.001
spacing = int((1 - density) + d_min)

c_sonicated = np.zeros(num_samples)
for i in range(num_samples):
	c_sonicated[i] = 

plt.figure()
plt.plot(r, density, '-', color='tab:orange')
plt.grid()
plt.show(block=False)

plt.figure()
plt.plot(c, 'o-')
plt.grid()
plt.show(block=False)

yay

F = fftn(f)
I_uc = np.abs(F)**2

C = fftn(c)
I_cryst = np.abs(C)**2

plt.figure()
plt.plot(I_cryst, '--')
plt.grid()
plt.show(block=False)
"""

"""
# Continuous density, non-crystalline, probablistic placement
r_max = 10
r_inc = 0.1

r = np.arange(0,r_max,r_inc)
num_r = len(r)

density = np.sin(r)

plt.figure()
plt.plot(r, d, '-', color='tab:orange')
plt.grid()
plt.show(block=False)


f = np.zeros(num_r)
"""






















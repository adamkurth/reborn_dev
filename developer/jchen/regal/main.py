"""
Testing strategies for iterative 

Date Created: 9 Aug 2022
Last Modified: 9 Aug 2022
Humans responsible: JC
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift, ifftshift

# Seed the random number generator
np.random.seed(43)

# Close all previous plots
plt.close('all')

# Plotting parameters
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
font_size = 14
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
CMAP = "viridis"



#=====================================================================
# Define some helper functions
def colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

def calc_MSE(x, x_true):
	return np.sum((np.abs(x - x_true))**2)

def see_Y(Y):
	# Visualise ys
	plt.figure()
	for k in range(K):
		plt.plot(fftshift(Y[:,k]))
	plt.grid()
	plt.xlabel('q')
	plt.show(block=False)

def see_C(C):
	# Visualise cs
	plt.figure()
	for k in range(K):
		plt.plot(C[k,:], 'o-')
	plt.grid()
	plt.xlabel('t')
	plt.show(block=False)

def calc_I(x):
	return np.dot(x['Y'],x['C'])

def make_an_initial_iterate():
	return {'Y': np.zeros((Q,K)), 'C': np.zeros((K,T))}


#=====================================================================
# Program parameters
Q = 500 # Number of q samples
T = 20  # Number of t samples
K = 4   # Number of states, or species

# qs = np.arange(0,)

it_max = 1000
starting_iterate = 1

beta_RAAR = 0.7

#=====================================================================

# Make y's

def make_ys(realspace_width):
	p = np.zeros(Q)
	p[0:realspace_width] = 1

	y = np.abs(fftn(p))**2
	y /= y[0]

	return y


y1 = make_ys(10)
y2 = make_ys(13)
y3 = make_ys(4)
y4 = make_ys(3)




plt.figure()
plt.plot(fftshift(y1))
plt.plot(fftshift(y2))
plt.plot(fftshift(y3))
plt.plot(fftshift(y4))
plt.grid()
plt.xlabel('q')
plt.show(block=False)




# Make c's
def make_cs(A, k, t0):
	ts = np.linspace(0,3,T)
	c1 = A/(1+ np.exp(-k*(ts-t0)))

	return c1


c1 = make_cs(A=1, k=1, t0=0)
c2 = make_cs(A=2, k=0.6, t0=0)
c3 = make_cs(A=0.3, k=2, t0=0)
c4 = make_cs(A=1.7, k=-0.7, t0=0)



plt.figure()
plt.plot(c1, 'o-')
plt.plot(c2, 'o-')
plt.plot(c3, 'o-')
plt.plot(c4, 'o-')
plt.grid()
plt.xlabel('t')
plt.show(block=False)



#=====================================================================
# Multiply together to give I

Y = np.zeros((Q,K))
Y[:,0] = y1
Y[:,1] = y2
Y[:,2] = y3
Y[:,3] = y4


C = np.zeros((K,T))
C[0,:] = c1
C[1,:] = c2
C[2,:] = c3
C[3,:] = c4

I = np.dot(Y,C)



plt.figure()
plt.plot(fftshift(I[:,0]), '-')
plt.plot(fftshift(I[:,int(1*T/4)]), '-') # Show the evolution of the trace in q space every quarter of the total measured timesteps
plt.plot(fftshift(I[:,int(2*T/4)]), '-')
plt.plot(fftshift(I[:,int(3*T/4)]), '-')
plt.plot(fftshift(I[:,-1]), '-')
plt.grid()
plt.xlabel('t')
plt.show(block=False)

plt.figure()
plt.imshow(fftshift(I.T, axes=1))
plt.show(block=False)

#-----------------------------
# Add noise, water ring etc.



#-----------------------------

I_data = I.copy()

#=====================================================================
# PM figure out

C_est, residuals, rank, sing_val = np.linalg.lstsq(Y, I, rcond=None)
print(np.sum(C_est - C))

Y_est, residuals, rank, sing_val = np.linalg.lstsq(C.T, I.T, rcond=None)
print(np.sum(Y_est.T - Y))




def P_M(x, M_in):
	# Unload content in M_in
	I_data = M_in['I_data']
	mask_I = M_in['mask_I']

	# Unload content in x
	Y0 = x['Y']
	C0 = x['C']

	C_est1, residuals, rank, sing_val = np.linalg.lstsq(Y0, I_data, rcond=None)
	Y_est_transpose1, residuals, rank, sing_val = np.linalg.lstsq(C_est1.T, I_data.T, rcond=None)

	Y_est_transpose2, residuals, rank, sing_val = np.linalg.lstsq(C0.T, I_data.T, rcond=None)
	C_est2, residuals, rank, sing_val = np.linalg.lstsq(Y_est_transpose2.T, I_data, rcond=None)


	C_est = (C_est1 + C_est2)/2
	Y_est_transpose = (Y_est_transpose1 + Y_est_transpose2)/2

	# C_est = C_est1
	# Y_est_transpose = Y_est_transpose1

	# C_est = C_est2
	# Y_est_transpose = Y_est_transpose2


	x_new = {'Y': Y_est_transpose.T, 'C': C_est}

	return x_new



#=====================================================================
# PS
# Leave C floating for now


# p_autocorr_est = ifftshift(ifftn(Y[:,3]))

# plt.figure()
# plt.plot(p_autocorr_est, 'o-')
# plt.title('autocorrelation of p(r), the pair distribution func')
# plt.show(block=False)


def P_S(x, S_in):
	# Unload content in S_in
	# supp = S_in['supp']
	# I_data = S_in['I_data']

	# Unload content in x
	Y0 = x['Y']
	C0 = x['C']

	# Enforce contraints on Y
	for k in range(K):
		p = 0.1
		p_k_autocorr_est = np.abs(ifftn(Y0[:,k]))
		p_k_autocorr_est[p_k_autocorr_est < p * np.max(p_k_autocorr_est)] = 0.0 # set everything less than p% of max to zero
		y_k_est = fftn(p_k_autocorr_est) # Modify in-place

		# y_k_est = Y0[:,k]

		# Real 
		y_k_est = np.real(y_k_est)

		# Positivity
		y_k_est[y_k_est < 0] = 0.0
		# y_k_est[y_k_est < 0] *= -1

		# Normalise to have maximum of 1
		y_k_est /= np.max(y_k_est)

		Y0[:,k] = y_k_est


	# Enforce contraints on C
	for k in range(K):
		# Make C's positive
		C0[k,:][C0[k,:]<0] = 0


	x_new = {'Y': Y0, 'C': C0}


	return x_new


#=====================================================================
# PM, PS testing

if 0:
	# Random start
	x = {'Y': np.random.rand(Q,K), 'C': np.random.rand(K,T)}


	# M and S dictionaries
	M_in = {'I_data': I_data, 'mask_I':0}
	S_in = {}

	x_new = P_M(x, M_in)

	see_Y(x_new['Y'])
	see_C(x_new['C'])



	x_new = P_S(x_new, S_in)

	see_Y(x_new['Y'])
	see_C(x_new['C'])


	x_new = P_M(x_new, M_in)

	see_Y(x_new['Y'])
	see_C(x_new['C'])



	yay



#=====================================================================
# IPA begin


def ER(x, S_in, M_in):
	x_PM = P_M(x, M_in)
	x_PS = P_S(x_PM, S_in)

	x_new = x_PS

	return x_new, x_PS, x_PM


# For RAAR
def raar_update(x, x_PM, x_PS, x_PSPM, beta):
	# Create storage for the modified iterate
	x_new = make_an_initial_iterate()

	x_new['Y'] = 2*beta * x_PSPM['Y'] \
		         - beta * x_PS['Y']   \
		         + (1 - 2*beta) * x_PM['Y']   \
		         + beta * x['Y']

	x_new['C'] = 2*beta * x_PSPM['C'] \
		         - beta * x_PS['C']   \
		         + (1 - 2*beta) * x_PM['C']   \
		         + beta * x['C']

	return x_new


def RAAR(x, S_in, M_in, beta_RAAR):
	x_PM = P_M(x, M_in)
	x_PS = P_S(x, S_in)
	x_PSPM = P_S(x_PM, S_in)

	x_new = raar_update(x, x_PM, x_PS, x_PSPM, beta_RAAR)

	return x_new, x_PS, x_PM 





#------------------
# Make a ground truth iterate for comparison
Y_true = Y.copy()
C_true = C.copy()
x_true = {'Y': Y_true, 'C': C_true}

#------------------
# Define inputs to projectors
M_in = {'I_data': I_data, 'mask_I':0}
S_in = {}

#------------------
# Initialise iterate
print("Initialising the starting iterate.")

if (starting_iterate == 0): # Ground truth (on the f) for testing
	print('Ground truth')
	x = {'Y': Y_true, 'C': C_true}

elif(starting_iterate == 1): # Random start
	print('Random start')
	x = {'Y': np.random.rand(Q,K), 'C': np.random.rand(K,T)}

elif(starting_iterate == 2): # Slight perturbation from Gnd truth
	print('Slight perturbation')
	perturb_mag = 0.01
	x = {'Y': Y_true + perturb_mag*np.random.rand(Q,K), 'C': C_true + perturb_mag*np.random.rand(K,T)}

#------------------
# Initialise error storage and error comparisons
errors = np.zeros((it_max, 3)) # 1 - e_I, 2 - e_Y, 3 - e_C
e_best = np.inf
it_best = 0
x_best = x.copy()
#------------------
# Calculate initial errors
I_est = calc_I(x)
errors[0,0] = calc_MSE(I_est, I_data)
errors[0,1] = calc_MSE(x['Y'], Y_true)
errors[0,2] = calc_MSE(x['C'], C_true)
#------------------


# Start of IPA loop
print("Start of IPA.")
for it in range(1, it_max):


	#-------------------------
	print('RAAR')
	x_new, x_PS, x_PM = RAAR(x, S_in, M_in, beta_RAAR)

	"""
	# Doing Andrew Martin's noise-tolerant algorithm.
	# Do both ER and IPA
	# x_new (the new iterate) is initialised to the output of IPA. Will then be modified by the mask calculated from thresholding P_M x.
	x_new_ER, x_PS_ER, x_PM_ER = ER(x, S_in, M_in)
	x_new   , x_PS_DM, x_PM_DM = DM(x, S_in, M_in, gamma_S, gamma_M, beta_DM)

	# Iteration-dependent mask
	x_PM_abs = np.abs(x_PM_ER['f'])
	mask_PM = (x_PM_abs > thresh_noiseIPA)

	mask_IPA = mask_PM * f_supp_c + f_supp
	mask_ER = (1 - mask_PM) * f_supp_c

	x_new['f'] = (x_new['f'] * mask_IPA) + (x_new_ER['f'] * mask_ER)
	"""
	#-------------------------
	# Error calculation
	x_est = x_new
	I_est = calc_I(x_est)

	errors[it,0] = calc_MSE(I_est, I_data)
	errors[it,1] = calc_MSE(x['Y'], Y_true)
	errors[it,2] = calc_MSE(x['C'], C_true)

	#-------------------------
	# Keep the best reconstruction so far as measured by the data error
	if errors[it,0] < e_best:
		e_best = errors[it,0]
		it_best = it
		x_best = x_est

	#-------------------------
	# Update iterate
	x = x_new
	#-------------------------

	# Print some outputs to monitor progress
	print("it=%d, e_I = %.2e, e_Y = %.2e, e_C = %.2e\n" % (it, errors[it,0], errors[it,1], errors[it,2]))

#----------------------
# Normalisations for the stored vectors
norming_const_0 = np.sum( I_data**2 )
norming_const_1 = np.sum( x_true['Y']**2 )
norming_const_2 = np.sum( x_true['C']**2 )

errors[:,0] = np.sqrt(errors[:,0] / norming_const_0)
errors[:,1] = np.sqrt(errors[:,1] / norming_const_1)
errors[:,2] = np.sqrt(errors[:,2] / norming_const_2)

#----------------------

print("Max error in E_PM = %.2e\n" % np.max(errors[:,0]))


#=====================================================================
# Results




# see_Y(x_best['Y'])
# see_C(x_best['C'])

# see_Y(x_true['Y'])
# see_C(x_true['C'])



# ax.plot(np.arange(0,it_max), np.log10(errors[:,0]), lw=2, color='b', linestyle='-', label='e')
# ax.plot(np.arange(0,it_max), np.log10(errors[:,1]), lw=2, color='r', linestyle='-', label='E')
# ax.set_xlim(0, it_max)
# # ax.set_ylim(top=1)
# ax.grid()
# # axes[0,0].xlabel('Iteration', fontsize=16)
# # axes[0,0].ylabel('Errors', fontsize=16)
# ax.legend(fontsize=font_size)




fig, axes = plt.subplots(2, 3, figsize=(8, 6))

ax = axes[0,0]
for k in range(K):
	ax.plot(fftshift(x_best['Y'][:,k]))
ax.grid()
ax.set_xlabel('q')
ax.set_title(f'Best Y, it={it_best:d}')


ax = axes[0,1]
for k in range(K):
	ax.plot(x_best['C'][k,:], 'o-')
ax.grid()
ax.set_xlabel('t')
ax.set_title(f'Best C')


ax = axes[1,0]
for k in range(K):
	ax.plot(fftshift(x_true['Y'][:,k]))
ax.grid()
ax.set_xlabel('q')
ax.set_title(f'True Y')


ax = axes[1,1]
for k in range(K):
	ax.plot(x_true['C'][k,:], 'o-')
ax.grid()
ax.set_xlabel('t')
ax.set_title(f'True C')


ax = axes[0,2]
ax.plot(errors[:,0], label='I')
ax.plot(errors[:,1], label='Y')
ax.plot(errors[:,2], label='C')
ax.grid()
ax.legend()
ax.set_title("Errors")


plt.tight_layout()
plt.show(block=False)



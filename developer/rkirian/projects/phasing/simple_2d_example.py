import numpy as np

np.random.seed(0)  # Initialize the random number generator so our results are reproducible

########## CONFIGURATIONS UPFRONT #################
n_pix = 128  # Powers of 2 are best for the Fast Fourier Transform algorithm (FFT)
###########################################


im = np.random.rand((n_pix/2, n_pix/2))

from reborn.analysis.peaks import boxsnr
import numpy as np
from time import time

shape = (3000, 3000)
dat = np.random.random(shape)
mask = np.ones(shape)
nin, ncent, nout = 1, 5, 10
t = time()
snr, sig = boxsnr(dat, mask, mask, nin, ncent, nout)
print(time()-t)

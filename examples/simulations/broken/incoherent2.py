import numpy as np
from bornagain.detector import PADGeometry
from bornagain.utils import vec_norm

n_patterns = 10
wavelength = 640e-9
beam_vec = np.array([0, 0, 1.0])
d = 100e-6
r = np.array([[d, 0, 0]])
n_atoms = r.shape[0]
pad = PADGeometry(n_pixels=512, pixel_size=6.5e-6, distance=0.01)
v = pad.position_vecs()
k = 2*np.pi/wavelength*(v/vec_norm(v))

for n in range(n_patterns):
    phase = np.random.rand()*np.pi*2
    intensity = 1 + np.exp(1j*np.dot(r, k) + phase)


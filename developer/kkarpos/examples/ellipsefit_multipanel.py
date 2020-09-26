r"""
Beam center from ellipse fit
============================

Simulate water scatter and then fit an ellipse to pixels associated to the water ring.

Contributed by Richard Kirian.

Edited by Kosta Karpos.

Note that this example
only works on single-panel detectors.  It could be extended to multi-panel detectors if the need arises, but in that
case it is important that the relative coordinates of all panels is known very well.

"""

# %%
# First we import the needed modules and configure the simulation parameters:

import numpy as np
from scipy import constants as const
import matplotlib.pyplot as plt
from reborn import source, detector
from reborn.analysis import optimize
from reborn.simulate import examples
# Need to be on reborn develop branch 
from reborn.simulate import solutions
from reborn.viewers.qtviews.padviews import PADView





r_e = const.value('classical electron radius')
eV = const.value('electron volt')

# Configure the detector
detector_shape = [200, 200]
beam_center = np.array([110, 105], dtype=np.double)
pixel_size = 750e-6
detector_distance = 0.1
# Configure x-ray beam
n_photons = 1e12
photon_energy = 8000*eV
beam_diameter = 5e-6
# Configure water
water_thickness = 1e-6
water_ring_thresh = 2000000

# %%
# Set up the PAD geometry:
# pad = detector.PADGeometry(distance=detector_distance, shape=detector_shape, pixel_size=pixel_size)


pad = detector.tiled_pad_geometry_list(pad_shape=(512, 1024), pixel_size=100e-6, distance=0.1, tiling_shape=(4, 2), pad_gap=5*100e-6)
px = np.concatenate([p.position_vecs()[:,0].ravel() for p in pad])
py = np.concatenate([p.position_vecs()[:,1].ravel() for p in pad])
# %%
# Set up the x-ray beam:
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=n_photons*photon_energy)

#%%
dat = solutions.get_pad_solution_intensity(pad, beam, thickness=10e-6, liquid='water')
data = detector.concat_pad_data(dat)
# viewer = PADView(pad_geometry=pad, raw_data=dat)
# viewer.start()

mask = data*0
print(mask.size)
w = np.where(data > 5000)
mask[w[0]] = 1
print(np.max(mask), np.sum(mask))
mask = detector.split_pad_data(pad, mask.ravel())
viewer = PADView(pad_geometry=pad, raw_data=mask)
viewer.start()


fit_ellipse = optimize.fit_ellipse(px[w], py[w])


# %%
# Fit an ellipse to pixel coordinates above a water-ring threshold:
#x, y = np.nonzero(dat > water_ring_thresh)
beam_center_fit = optimize.ellipse_center(fit_ellipse)
# %%
# Display results:
print('True beam center:', beam_center)
print('Beam center from ellipse fit:', beam_center_fit)
print('Fractional errors:', np.abs(beam_center-beam_center_fit)/beam_center)
# plt.imshow(beam_center_fit, cmap='gray')
# plt.title('Water scatter (only pixels above threshold)')
plt.show()
import numpy as np
import reborn
from reborn.viewers.qtviews import PADView
from reborn.simulate.solutions import get_pad_solution_intensity
from reborn.const import eV
# X-ray pixel-array detector (PAD) geometry.  XFEL detectors are segmented into several "ASICs".
geom = reborn.detector.jungfrau4m_pad_geometry_list(detector_distance=0.1)
# X-ray beam properties
beam = reborn.source.Beam(photon_energy=9500*eV, diameter_fwhm=1e-6, pulse_energy=1e-3)
# No analysis routine is useful if it doesn't make use of a bad-pixel mask:
mask = geom.edge_mask()
# This simulates a water pattern
pattern = get_pad_solution_intensity(pad_geometry=geom, thickness=3e-6, beam=beam, poisson=True)
# For convenience, concatenate the PAD data into one contiguous array
pattern = geom.concat_data(pattern)
# Mess up edge pixels to emulate some "real-world" problems
a = pattern[geom.concat_data(mask) == 0]
pattern[geom.concat_data(mask) == 0] = a*0 + np.random.rand(len(a))*20
# Now make a phony jet streak.  This is not a physical model; just some hacking...
angle = (90-15)*np.pi/180  # Angle of the streak
# Some fancy footwork with vectors...
pixel_vecs = geom.s_vecs()  # Unit vectors from origin to detector pixels
beam_vec = beam.beam_vec  # Incident beam unit vector
# Make the unit vector that defines the streak.  The x-ray E-field vectors are a convenient basis.
streak_vec = beam.e1_vec * np.cos(angle) + beam.e2_vec * np.sin(angle)
# Angular deviation from plane of the streak
phi = 90*np.pi/180 - np.arccos(np.abs(np.dot(pixel_vecs, streak_vec)))
# Angular deviation from x-ray beam
theta = np.arccos(np.abs(np.dot(pixel_vecs, beam.beam_vec)))
# Fiddle with the numbers to make it look like a jet streak.
streak = np.random.poisson(100*np.exp(-1000000*phi**2 - 100*theta**2))
pattern += streak
# Have a look
padview = PADView(data=pattern, pad_geometry=geom, beam=beam)
padview.start()
# Now mask the streak
smask = geom.streak_mask(vec=streak_vec, angle=0.01)
padview = PADView(data=pattern*smask, pad_geometry=geom, beam=beam)
padview.start()

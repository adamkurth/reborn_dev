import numpy as np
import matplotlib.pyplot as plt
from bornagain.external import crystfel
from bornagain.data import cspad_geom_file

# This converts a CrytFEL geom file to a *list* of PADGeometry instances
pads = crystfel.geometry_file_to_pad_geometry_list(cspad_geom_file)
# For each PADGeometry, we get the q vectors
q_vecs = []
for p in pads:
    q_vecs.append(p.q_vecs(wavelength=1.5e-10, beam_vec=[0, 0, 1]))
# Scatter plot the x-y coordinates
for q in q_vecs:
    plt.scatter(q[::10, 0], q[::10, 1])
plt.show()
# Similarly, you can get the q magnitudes:
q_mags = [p.q_mags(wavelength=1.5e-10, beam_vec=[0, 0, 1]) for p in pads]

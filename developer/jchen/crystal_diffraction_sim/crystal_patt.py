"""
Script to simulate crystal diffraction pattern with beam divergence.

Created: 11 Sep 2021
Last Modified: 12 Sep 2021
Humans responsible: JC, RAK
"""

from reborn import detector,source
from reborn.simulate.examples import CrystalSimulatorV1
from reborn.target import crystal

import numpy as np
#pads = detector.cspad_2x2_pad_geometry()

pads = detector.cspad_pad_geometry_list()
beam = source.Beam(wavelength=1.5e-10, diameter_fwhm=1e-6)
beam.beam_divergence_fwhm = 1*np.pi/180
beam.photon_energy_fwhm = beam.photon_energy * 0.01 

cryst = crystal.CrystalStructure('2LYZ')
cryst.mosaic_domain_size = 200e-9
cryst.mosaic_domain_size_fwhm = 50e-9
cryst.mosaicity_fwhm = 3/180 * np.pi
cryst.crystal_size = 10e-6
cryst.crystal_size_fwhm = 5e-6

a = CrystalSimulatorV1(pad_geometry=pads, beam=beam, crystal_structure=cryst, n_iterations=5,
                 approximate_shape_transform=True, expand_symmetry=True,
                 cl_double_precision=False, cl_group_size=32, poisson_noise=False)

pattern = a.generate_pattern()

from reborn.viewers.qtviews import PADView 

pv = PADView(raw_data=pattern, pad_geometry=pads)
pv.start()


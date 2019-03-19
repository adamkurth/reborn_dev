#!/usr/bin/env python

import numpy as np

from bornagain import detector, source, target
from bornagain.simulate.examples import CrystalSimulatorV1
from bornagain.simulate import examples
from bornagain.units import keV
from bornagain.viewers.qtviews.padviews import PADView
from bornagain.fileio.getters import FrameGetter

detector_distance = 50e-3
pixel_size = 75e-6
n_pixels = 1000
beam_diameter = 10e-6
n_photons = 1e12
mosaicity_fwhm = 0.0
beam_divergence_fwhm = 0.01
beam_spatial_profile = 'tophat'
photon_energy_fwhm = 0.0
crystal_size = 10e-6
crystal_size_fwhm = 0.0
mosaic_domain_size = 1e-6
mosaic_domain_size_fwhm = 0.0
pdb_file = examples.lysozyme_pdb_file

pad = detector.PADGeometry(n_pixels=n_pixels, pixel_size=pixel_size, distance=detector_distance)

beam = source.Beam(photon_energy=9.0 / keV)
beam.photon_energy_fwhm = beam.photon_energy * 0.001
beam.diameter_fwhm = 10e-6
beam.beam_divergence_fwhm = 0.0001
beam.pulse_energy = beam.photon_energy * 1e10

cryst = target.crystal.CrystalStructure(pdb_file)
cryst.crystal_size = 10e-6
cryst.mosaic_domain_size = 1e-6
cryst.crystal_size_fwhm = cryst.crystal_size * 0.001
cryst.mosaic_domain_size_fwhm = cryst.mosaic_domain_size * 0.001

simulator = CrystalSimulatorV1(pad_geometry=pad, beam=beam, crystal_structure=cryst, n_iterations=100,
                               random_rotation=True, approximate_shape_transform=True, cromer_mann=False,
                               expand_symmetry=False, cl_double_precision=False, cl_group_size=32, poisson_noise=True)


class MyFrameGetter(FrameGetter):
    generate_pattern = None

    def get_frame(self, frame_number=1):
        intensities = self.generate_pattern()
        print(intensities)
        print(np.isfinite(intensities).all())
        print(intensities.min(), intensities.max())
        return {'pad_data': [intensities]}


frame_getter = MyFrameGetter()
frame_getter.generate_pattern = simulator.generate_pattern

padview = PADView(pad_geometry=[pad], frame_getter=frame_getter)
padview.start()

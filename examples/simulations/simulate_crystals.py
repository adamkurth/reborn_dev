#!/usr/bin/env python

from bornagain import detector, source, target
from bornagain.simulate.examples import CrystalSimulatorV1
from bornagain.simulate import examples
from bornagain.units import keV
from bornagain.viewers.qtviews.padviews import PADView
from bornagain.fileio.getters import FrameGetter

detector_distance = 50e-3
pixel_size = 110e-6
n_pixels = 1000
beam_diameter = 1e-6
photon_energy = 9.0 / keV
n_photons = 1e12
mosaicity_fwhm = 0e-4
beam_divergence_fwhm = 0e-2
beam_spatial_profile = 'tophat'
photon_energy_fwhm = 0.0
crystal_size = 1e-6
crystal_size_fwhm = 0e-6
mosaic_domain_size = 0.1e-6
mosaic_domain_size_fwhm = 0.0
water_radius = 0.0
temperature = 298.16
n_monte_carlo_iterations = 1
num_patterns = 1
random_rotation = True
approximate_shape_transform = True
cromer_mann = False
expand_symm = False
fix_rot_seq = False
pdb_file = examples.lysozyme_pdb_file
write_hdf5 = True
write_geom = True
results_dir = './temp'
quiet = False
compression = None
cl_double_precision = False

pad = detector.PADGeometry(n_pixels=n_pixels, pixel_size=pixel_size, distance=detector_distance)

beam = source.Beam(photon_energy=photon_energy)
beam.photon_energy_fwhm = photon_energy_fwhm
beam.diameter_fwhm = beam_diameter
beam.beam_divergence_fwhm = beam_divergence_fwhm
beam.pulse_energy = n_photons * photon_energy

cryst = target.crystal.CrystalStructure(pdb_file)
cryst.crystal_size = crystal_size
cryst.mosaic_domain_size = mosaic_domain_size
cryst.crystal_size_fwhm = crystal_size * crystal_size_fwhm
cryst.mosaic_domain_size_fwhm = mosaic_domain_size * mosaic_domain_size_fwhm

simulator = CrystalSimulatorV1(pad_geometry=pad, beam=beam, crystal_structure=cryst, n_iterations=1,
                               random_rotation=True, approximate_shape_transform=True, cromer_mann=False,
                               expand_symmetry=False, cl_double_precision=False, cl_group_size=32, poisson_noise=True)


class MyFrameGetter(FrameGetter):
    generate_pattern = None

    def get_frame(self, frame_number=1):
        return {'raw_data': [self.generate_pattern()]}


frame_getter = MyFrameGetter()
frame_getter.generate_pattern = simulator.generate_pattern

padview = PADView(pad_geometry=[pad], frame_getter=frame_getter)
padview.start()

from bornagain import detector, source, target
from bornagain.simulate.examples import CrystalSimulatorV1
from bornagain.simulate import examples
from bornagain.units import keV
from bornagain.viewers.qtviews.padviews import PADView
from bornagain.fileio.getters import FrameGetter

pad = detector.PADGeometry(n_pixels=1000, pixel_size=75e-6, distance=50e-3)

beam = source.Beam(photon_energy=9.0 / keV)
beam.photon_energy_fwhm = beam.photon_energy * 0.001
beam.diameter_fwhm = 10e-6
beam.beam_divergence_fwhm = 0.0001
beam.pulse_energy = beam.photon_energy * 1e8

cryst = target.crystal.CrystalStructure(examples.lysozyme_pdb_file)
cryst.crystal_size = 10e-6
cryst.mosaic_domain_size = 0.5e-6
cryst.crystal_size_fwhm = cryst.crystal_size * 0.001
cryst.mosaic_domain_size_fwhm = cryst.mosaic_domain_size * 0.001
cryst.mosaicity_fwhm = 0.01

simulator = CrystalSimulatorV1(pad_geometry=pad, beam=beam, crystal_structure=cryst, n_iterations=100,
                               approximate_shape_transform=True, cromer_mann=False, expand_symmetry=False,
                               cl_double_precision=False, cl_group_size=32, poisson_noise=True)


class MyFrameGetter(FrameGetter):
    generate_pattern = None

    def get_frame(self, frame_number=1):
        return {'pad_data': [self.generate_pattern()]}


frame_getter = MyFrameGetter()
frame_getter.generate_pattern = simulator.generate_pattern

padview = PADView(pad_geometry=[pad], frame_getter=frame_getter)
padview.start()

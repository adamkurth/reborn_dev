from bornagain import detector, source, target
from bornagain.simulate.examples import CrystalSimulatorV1
from bornagain.simulate import examples
from bornagain.viewers.qtviews.padviews import PADView
from bornagain.fileio.getters import FrameGetter
from scipy import constants

eV = constants.value('electron volt')

pad = detector.PADGeometry(n_pixels=512, pixel_size=200e-6, distance=100e-3)

beam = source.Beam(photon_energy=2000 * eV)
beam.photon_energy_fwhm = beam.photon_energy * 0.001
beam.diameter_fwhm = 1e-6
beam.beam_divergence_fwhm = 0.0001
beam.pulse_energy = beam.photon_energy * 1e12

cryst = target.crystal.CrystalStructure(examples.lysozyme_pdb_file)
cryst.crystal_size = 0.2e-6
cryst.mosaic_domain_size = 0.2e-6
cryst.crystal_size_fwhm = cryst.crystal_size * 0.001
cryst.mosaic_domain_size_fwhm = cryst.mosaic_domain_size * 0.001
cryst.mosaicity_fwhm = 0.01

simulator = CrystalSimulatorV1(pad_geometry=pad, beam=beam, crystal_structure=cryst, n_iterations=1,
                               approximate_shape_transform=False, cromer_mann=False, expand_symmetry=False,
                               cl_double_precision=False, cl_group_size=32, poisson_noise=True)

simulator.generate_pattern()

class MyFrameGetter(FrameGetter):
    generate_pattern = None

    def get_frame(self, frame_number=1):
        dat = {'pad_data': [self.generate_pattern()]}
        return dat


frame_getter = MyFrameGetter()
frame_getter.generate_pattern = simulator.generate_pattern

padview = PADView(pad_geometry=[pad], frame_getter=frame_getter)
padview.start()

import numpy as np
from time import time
from scipy import constants
from reborn import detector, source, target
from reborn.simulate.examples import CrystalSimulatorV1, cspad_geom_file, lysozyme_pdb_file
from reborn.external.crystfel import geometry_file_to_pad_geometry_list
from reborn.viewers.qtviews.padviews import PADView
from reborn.fileio.getters import FrameGetter

# Everything in bornagain is SI units.
eV = constants.value('electron volt')

# Load a CrystFEL geom file
pads = geometry_file_to_pad_geometry_list(cspad_geom_file)
# We need to set the detector distance, i.e. "z" component of the detector translation vector:
for p in pads:
    p.t_vec[2] = 0.2

# Set up the properties of the x-ray beam.  The beam direction is along the "z" axis unless otherwise specified.
beam = source.Beam(photon_energy=9000*eV)
beam.photon_energy_fwhm = beam.photon_energy * 0.001
beam.diameter_fwhm = 10e-6
beam.beam_divergence_fwhm = 0.005
beam.pulse_energy = beam.photon_energy * 1e10

# Set up the properties of the crystal we are shooting.
cryst = target.crystal.CrystalStructure(lysozyme_pdb_file)
cryst.crystal_size = 10e-6
cryst.mosaic_domain_size = 0.5e-6
cryst.crystal_size_fwhm = cryst.crystal_size * 0.1
cryst.mosaic_domain_size_fwhm = cryst.mosaic_domain_size * 0.1
cryst.mosaicity_fwhm = 0.01

# Since we are dealing with multi-panel detectors, our masks, dark current, intensites, and so on are all lists of
# 2D numpy arrays.
masks = [p.beamstop_mask(q_min=2*np.pi/500e-10, beam=beam) for p in pads]

# Do not rely on CrystalSimulatorV1 -- it will likely change in the near future.
simulator = CrystalSimulatorV1(pad_geometry=pads, beam=beam, crystal_structure=cryst, n_iterations=1000,
                               approximate_shape_transform=True, expand_symmetry=False,
                               cl_double_precision=False, cl_group_size=32, poisson_noise=True)


# FrameGetter is a class that helps create a unified interface for serving up XFEL events.  The underlying code could
# read from a CXIDB file, an XTC file, from shared memory, or whatever else is convenient.  For this example, our
# FrameGetter subclass will generate simulations on the fly.  Making a FrameGetter sub-class is easy: we just need to
# override one method called "get_data(frame_number)".
class MyFrameGetter(FrameGetter):
    def get_data(self, frame_number=1):
        t = time()
        dat = {'pad_data': simulator.generate_pattern()}
        print('Simulation in %g seconds' % (time()-t,))
        return dat


frame_getter = MyFrameGetter()

# PADView is a very basic viewer that is in development.  It can link up with a FrameGetter in order to serve up frames
# from various file formats, or in this case it can be used to look at simulations one-by-one.
padview = PADView(pad_geometry=pads, frame_getter=frame_getter, mask_data=masks)
padview.set_levels(-1, 10)
padview.start()

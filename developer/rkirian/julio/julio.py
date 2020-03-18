from xyz import xyz_reader
import numpy as np


# file_name = "Ar4631.xyz"
file_name = "Ar_1000.xyz"
xyz = xyz_reader(file_name)

atomic_numbers = xyz[0, :].astype(np.int)
r_vecs = xyz[1:, :].T/1e10

deluxe = True

if not deluxe:

    #####################################################################
    # This chunk simply gets the q-vectors
    ###################################################################

    from bornagain.detector import PADGeometry

    # Everything in bornagain is SI units
    detector_distance = 0.07
    pixel_size = 110e-6
    beam_direction = [0, 0, 1]
    n_pixels = 256

    # Create an instance of a bornagain pixel-array detector (PAD) geometry class
    pad = PADGeometry(shape=[n_pixels, n_pixels], pixel_size=pixel_size, distance=detector_distance)
    q_vecs = pad.q_vecs(beam=beam_direction)
    # If the only thing you need are q-vectors (2*pi/lambda) then we are done.  The above is an N^2x3 array, where N is
    # the number of pixels along the edge of the (square) detector.

if deluxe:

    ########################################################################
    # Below is the deluxe version
    ########################################################################

    import numpy as np
    from bornagain.detector import PADGeometry
    from bornagain.source import Beam
    from bornagain.target.crystal import CrystalStructure
    from bornagain.viewers.qtviews import PADView
    from bornagain.simulate.atoms import xraylib_scattering_factors
    from bornagain.simulate.clcore import ClCore
    import scipy.constants as const

    eV = const.value('electron volt')
    r_e = const.value('classical electron radius')

    # Everything in bornagain is SI units
    photon_energy = 9000 * eV  # Joules
    detector_distance = 0.03  # Meters
    pulse_energy = 200e-3  # Joules
    pixel_size = 110e-6  # Meters
    beam_direction = [0, 0, 1]  # Incident beam direction
    n_pixels = 256

    # This is the x-ray beam information.  Most importantly, it contains incident fluence and direction of the beam.
    beam = Beam(photon_energy=photon_energy, diameter_fwhm=0.2e-6, pulse_energy=pulse_energy)
    fluence = beam.photon_number_fluence

    # Create an instance of a pixel-array detector (PAD) geometry class
    pad = PADGeometry(shape=[n_pixels, n_pixels], pixel_size=pixel_size, distance=detector_distance)
    q_vecs = pad.q_vecs(beam=beam)
    q_mags = pad.q_mags(beam=beam)
    polarization_factors = pad.polarization_factors(beam=beam)
    solid_angles = pad.solid_angles()

    # Gather atomic coordinates etc. from a PDB file
    pdb_id = '1jb0'
    cryst = CrystalStructure(pdb_id)
    # r_vecs = cryst.molecule.coordinates
    # atomic_numbers = cryst.molecule.atomic_numbers
    # We will group atoms with the same atomic number since they have a common form factor
    uniq_z = np.unique(atomic_numbers)
    grp_r_vecs = []
    grp_fs = []
    for z in uniq_z:
        subr = np.squeeze(r_vecs[np.where(atomic_numbers == z), :])
        grp_r_vecs.append(subr)
        grp_fs.append(xraylib_scattering_factors(q_mags, photon_energy=beam.photon_energy, atomic_number=z))

    # This creates an instance of the OpenCL GPU simulation engine, which manages GPU context etc.
    clcore = ClCore()
    # This moves the q-vectors to the GPU device.  Do this only once since it is a costly memory operation.
    q_vecs_gpu = clcore.to_device(q_vecs)

    R = np.eye(3)
    print('Simulating pattern...')
    amps = 0
    for j in range(len(uniq_z)):
        f = grp_fs[j]
        r = grp_r_vecs[j]
        a = clcore.phase_factor_qrf(q_vecs_gpu, r, R=R)  # GPU sum over f(q)*exp(i Rq.r), where f(q) is 1 in this case
        amps += a*f
    intensities = r_e**2*fluence*solid_angles*polarization_factors*np.abs(amps)**2  #*np.abs(fs[i])**2
    # intensities = np.random.poisson(intensities)
    intensities = pad.reshape(intensities)  # Make it a 2D array for display purposes

    padview = PADView(raw_data=intensities, pad_geometry=pad)
    padview.set_levels(-0.1, 10)
    padview.start()
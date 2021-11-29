import numpy as np
import scipy.constants as const
from scipy.spatial.transform import Rotation
from reborn import detector
from reborn.source import Beam
from reborn.target import crystal, atoms
from reborn.simulate import clcore
from reborn.viewers.mplviews import view_pad_data

eV = const.value('electron volt')
r_e = const.value('classical electron radius')
np.random.seed(42)  # Make random numbers that are reproducible

beam = Beam(photon_energy=8000*eV, diameter_fwhm=0.2e-6, pulse_energy=1e-3)
fluence = beam.photon_number_fluence
pads = detector.jungfrau4m_pad_geometry_list(detector_distance=0.36)
q_vecs = [pad.q_vecs(beam=beam) for pad in pads]
solid_angles = [pad.solid_angles() for pad in pads]
polarization_factors = [pad.polarization_factors(beam=beam) for pad in pads]
q_mags = [p.q_mags(beam=beam) for p in pads]

qmin = np.min(np.array([np.min(q) for q in q_mags]))
qmax = np.max(np.array([np.max(q) for q in q_mags]))
print('resolution range (Angstrom): ', 2*np.pi*1e10/qmax, ' - ', 2*np.pi*1e10/qmin)

cryst = crystal.CrystalStructure('2LYZ.pdb')#('BDNA25_sp_mod.pdb')  #('2LYZ.pdb')
r_vecs = cryst.molecule.coordinates  # Atomic coordinates of the asymmetric unit
r_vecs -= np.mean(r_vecs, axis=0)  # Roughly center the molecule
atomic_numbers = cryst.molecule.atomic_numbers

simcore = clcore.ClCore()

simcore.print_device_info()


uniq_z = np.unique(atomic_numbers)
grouped_r_vecs = []
grouped_fs = []
for z in uniq_z:
    subr = np.squeeze(r_vecs[np.where(atomic_numbers == z), :])
    grouped_r_vecs.append(subr)
    grouped_fs.append(atoms.hubbel_henke_scattering_factors(q_mags=q_mags, photon_energy=beam.photon_energy,
                                                            atomic_number=z))


thet = 90/180 * np.pi
c = np.cos(thet)
s = np.sin(thet)
R = np.array([[c, 0, s], [0, 1 ,0], [-s, 0 , c]])#np.eye(3)#Rotation.random().as_matrix()  # Just for fun, let's rotate the molecule

# from reborn.utils.

print('simulating intensities')
intensities = []
for i in range(len(pads)):
    print(f'simulating pad number {i}')
    pad = pads[i]
    sa = solid_angles[i]
    p = polarization_factors[i]
    q = q_vecs[i]
    amps = 0
    for j in range(len(grouped_fs)):
        f = grouped_fs[j][i]
        r = grouped_r_vecs[j]
        a = simcore.phase_factor_qrf(q, r, R=R)
        amps += a*f
    ints = r_e**2*fluence*sa*p*np.abs(amps)**2
    intensities.append(pad.reshape(ints))



print('# photons total: %d' % np.round(np.sum(detector.concat_pad_data(intensities))))

dispim = [np.log10(d+0) for d in intensities]
view_pad_data(pad_data=dispim, pad_geometry=pads)

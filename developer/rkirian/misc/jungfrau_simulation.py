r"""
Molecule diffraction from a PDB file
====================================

Contributed by Richard A. Kirian.

Imports:
"""

import numpy as np
import scipy.constants as const
from scipy.spatial.transform import Rotation
from reborn.simulate.examples import jungfrau4m_pads
from reborn.source import Beam
from reborn.target.crystal import pdb_to_dict, get_pdb_file
from reborn.simulate.atoms import xraylib_scattering_factors, atomic_symbols_to_numbers
from reborn.simulate.clcore import ClCore
from reborn.utils import max_pair_distance
from reborn.viewers.mplviews import view_pad_data

# %%
# Define some constants and other parameters that we'll need:
eV = const.value('electron volt')
r_e = const.value('classical electron radius')
np.random.seed(0)  # Make random numbers that are reproducible
photon_energy = 9000*eV
detector_distance = 0.2
pulse_energy = 2
pdb_id = '2LYZ'  # This is Lysozyme

# %%
# Setup the beam and detector:

beam = Beam(photon_energy=photon_energy, diameter_fwhm=0.2e-6, pulse_energy=pulse_energy)
fluence = beam.photon_number_fluence
binning = 16
pads = jungfrau4m_pads(detector_distance=detector_distance, binning=binning)
q_vecs = [pad.q_vecs(beam=beam) for pad in pads]
solid_angles = [pad.solid_angles() for pad in pads]
polarization_factors = [pad.polarization_factors(beam=beam) for pad in pads]
qmags = [p.q_mags(beam=beam) for p in pads]
qmin = np.min(np.array([np.min(q) for q in qmags]))
qmax = np.max(np.array([np.max(q) for q in qmags]))
print('resolution (A): ', 2*np.pi*1e10/qmax)

# %%
# Here we load a lysozyme PDB file, which is included in reborn for demonstrations like this.  You can try a different
# pdb ID; the file will be downloaded if it is not cached in the reborn git repository.
pdb_file = get_pdb_file(pdb_id)

# %%
# Here we convert the pdb file to a python dictionary.  This is the *only* piece of code in reborn that is not in SI
# units, and the reason for this is because the pdb_to_dict function is meant to directly convert a PDB file to a
# dictionary.  Most likely, this example will change in the future to avoid the appearance of non-SI units.
pdb_dict = pdb_to_dict(pdb_file)
r_vecs = pdb_dict['atomic_coordinates']*1e-10  # Atomic coordinates of the asymmetric unit
r_vecs -= np.mean(r_vecs, axis=0)  # Roughly center the molecule
atomic_numbers = atomic_symbols_to_numbers(pdb_dict['atomic_symbols'])

# %%
# Here we create a ClCore instance.  In reborn, the ClCore class helps maintain a context and queue with a GPU device.
# It has the functions you probably need in order to do simulations.
sim = ClCore()

# %%
# Let's see what kind of device we are running on.  If it is not a GPU, the simulations will not be very fast...
print(sim)


uniq_z = np.unique(atomic_numbers)
grp_r_vecs = []
grp_fs = []
for z in uniq_z:
    subr = np.squeeze(r_vecs[np.where(atomic_numbers == z), :])
    grp_r_vecs.append(subr)
    print(z, subr.shape)
    grp_fs.append([xraylib_scattering_factors(q, photon_energy=beam.photon_energy, atomic_number=z) for q in qmags])

print('Overall size', max_pair_distance(r_vecs))

n_patterns = 1
R = Rotation.random().as_matrix()
intensities = []
for pat in range(n_patterns):
    print('pattern %d' % pat)
    for i in range(len(pads)):
        pad = pads[i]
        sa = solid_angles[i]
        p = polarization_factors[i]
        q = q_vecs[i]
        amps = 0
        for j in range(len(grp_fs)):
            # print('panel', i, ', z =', uniq_z[j])
            f = grp_fs[j][i]
            r = grp_r_vecs[j]
            a = sim.phase_factor_qrf(q, r, R=R)
            amps += a*f
            # amps += a*uniq_z[j]
        ints = r_e**2*fluence*sa*p*np.abs(amps)**2  #*np.abs(fs[i])**2
        ints = np.random.poisson(ints)
        if pat == 0:
            intensities.append(pad.reshape(ints))
        else:
            intensities[i] += pad.reshape(ints)

for i in range(len(intensities)):
    print(type(intensities[i]))
    print(type(n_patterns))
    intensities[i] = intensities[i]/float(n_patterns)

print('# photons total: %d' % np.round(np.sum(np.ravel(intensities))))

# scat = [(np.abs(fs[i])**2).reshape(pads[i].shape()) for i in range(len(pads))]
# qmags =


dispim = [np.log10(d+1) for d in intensities]
view_pad_data(pad_data=dispim, pad_geometry=pads)

# dat = [pads[i].reshape(np.abs(grp_fs[3][i])**2) for i in range(len(pads))]
# dat = [np.reshape(qmags[i], pads[i].shape()) for i in range(len(pads))]
# padview = PADView(raw_data=dat, pad_geometry=pads)
# padview.set_levels(-0.2, 2)
# # padview.show_all_geom_info()
# padview.start()
# print('done')

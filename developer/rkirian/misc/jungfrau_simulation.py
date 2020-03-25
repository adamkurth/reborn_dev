import numpy as np
import matplotlib.pyplot as plt
import xraylib
import pyqtgraph as pg
from reborn.detector import tiled_pad_geometry_list
from reborn.source import Beam
from reborn.target.crystal import FiniteLattice, UnitCell, CrystalStructure, pdb_to_dict
from reborn.viewers.qtviews import PADView, Scatter3D
from reborn.simulate.atoms import xraylib_scattering_factors, atomic_symbols_to_numbers
from reborn.simulate.clcore import ClCore
from reborn.utils import max_pair_distance
import scipy.constants as const
from scipy.spatial.transform import Rotation

eV = const.value('electron volt')
r_e = const.value('classical electron radius')


photon_energy = 9000*eV
distance = 0.07
pulse_energy = 2e-3

beam = Beam(photon_energy=photon_energy, diameter_fwhm=0.2e-6, pulse_energy=pulse_energy)

# Construct the CXI Jungfrau 4M detector, made up of 8 modules arranged around a 9mm beamhole.  The number of pixels per
# module is 1024 x 512 and the pixel size is 75 microns.
bin = 16
pads = tiled_pad_geometry_list(pad_shape=(int(512/bin), int(1024/bin)), pixel_size=75e-6*bin, distance=distance,
                               tiling_shape=(4, 2), pad_gap=36*75e-6)
gap = 9e-3
pads[0].t_vec += + np.array([1, 0, 0])*gap/2 - np.array([0, 1, 0])*gap/2
pads[1].t_vec += + np.array([1, 0, 0])*gap/2 - np.array([0, 1, 0])*gap/2
pads[2].t_vec += - np.array([1, 0, 0])*gap/2 - np.array([0, 1, 0])*gap/2
pads[3].t_vec += - np.array([1, 0, 0])*gap/2 - np.array([0, 1, 0])*gap/2
pads[4].t_vec += + np.array([1, 0, 0])*gap/2 + np.array([0, 1, 0])*gap/2
pads[5].t_vec += + np.array([1, 0, 0])*gap/2 + np.array([0, 1, 0])*gap/2
pads[6].t_vec += - np.array([1, 0, 0])*gap/2 + np.array([0, 1, 0])*gap/2
pads[7].t_vec += - np.array([1, 0, 0])*gap/2 + np.array([0, 1, 0])*gap/2
q_vecs = [pad.q_vecs(beam=beam) for pad in pads]
solid_angles = [pad.solid_angles() for pad in pads]
polarization_factors = [pad.polarization_factors(beam=beam) for pad in pads]


pdb_dict = pdb_to_dict('../data/thiolGold_.pdb')
r_vecs = pdb_dict['atomic_coordinates']*1e-10
r_vecs -= np.mean(r_vecs, axis=0)
atomic_numbers = atomic_symbols_to_numbers(pdb_dict['atomic_symbols'])

# scat = Scatter3D()
# scat.add_points(r_vecs)
# scat.show()

# Construct a gold nanocluster
# dspace = 0.3e-9
# cell = UnitCell(dspace, dspace, dspace, np.pi/2, np.pi/2, np.pi/2)

# cryst = CrystalStructure('data/shelxpro.pdb')
# lat = FiniteLattice(max_size=50, unitcell=cell)
# lat.sphericalize(1.1e-9)
# r_vecs = lat.occupied_r_coordinates.copy()
# r_vecs = cryst.molecule.coordinates.copy()
# z = cryst.molecule.atomic_numbers.copy()
# r_vecs = np.compress(z == 79, r_vecs, axis=0)
# Atomic scattering factors
qmags = [p.q_mags(beam=beam) for p in pads]
qmin = np.min(np.array([np.min(q) for q in qmags]))
qmax = np.max(np.array([np.max(q) for q in qmags]))
print('resolution (A): ', 2*np.pi*1e10/qmax)
# qprof = np.arange(qmin, qmax, (qmax-qmin)/1000.)
# fprof = xraylib_scattering_factors(qmags=qprof, photon_energy=beam.photon_energy, atomic_number=79)
# plt.plot(qprof, np.abs(fprof))
# plt.show()
# print('getting scattering factors')
# fs = [xraylib_scattering_factors(q, photon_energy=beam.photon_energy, atomic_number=79) for q in qmags]
# print('done')

fluence = beam.photon_number_fluence
sim = ClCore()
uniq_z = np.unique(atomic_numbers)
grp_r_vecs = []
grp_fs = []
for z in uniq_z:
    subr = np.squeeze(r_vecs[np.where(atomic_numbers == z), :])
    grp_r_vecs.append(subr)
    print(z, subr.shape)
    grp_fs.append([xraylib_scattering_factors(q, photon_energy=beam.photon_energy, atomic_number=z) for q in qmags])

print('Overall size', max_pair_distance(r_vecs))
print('N gold', grp_r_vecs[np.where(uniq_z == 79)[0][0]].shape[0])
print('Gold size', max_pair_distance(grp_r_vecs[np.where(uniq_z == 79)[0][0]]))

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


dat = intensities
# dat = [pads[i].reshape(np.abs(grp_fs[3][i])**2) for i in range(len(pads))]
# dat = [np.reshape(qmags[i], pads[i].shape()) for i in range(len(pads))]
padview = PADView(raw_data=dat, pad_geometry=pads)
padview.set_levels(-0.2, 2)
# padview.show_all_geom_info()
padview.start()
print('done')

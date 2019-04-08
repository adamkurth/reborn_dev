import sys
import numpy as np
from numpy import exp, log
from scipy.misc import factorial
import pyqtgraph as pg
from bornagain.detector import PADGeometry
from bornagain.source import Beam
from bornagain.target.molecule import Molecule
from bornagain.target.density import trilinear_insertion
from bornagain.simulate.clcore import ClCore
from bornagain.simulate.examples import lysozyme_pdb_file, MoleculeSimulatorV1
from bornagain.utils import rotate, random_rotation, max_pair_distance
from bornagain.units import r_e

real_t = np.float64
clcore = ClCore(group_size=32)
beam = Beam(wavelength=3e-10, pulse_energy=100e-3, diameter_fwhm=100e-9)
pad = PADGeometry(n_pixels=1000, pixel_size=100e-6, distance=0.1)
mol = Molecule(pdb_file=lysozyme_pdb_file)
sim = MoleculeSimulatorV1(clcore=clcore, beam=beam, pad=pad, molecule=mol)

q_vecs = pad.q_vecs(beam=beam)
f = mol.get_scattering_factors(beam=beam)
intensity_prefactor = pad.reshape(beam.photon_number_fluence * r_e**2 * pad.solid_angles() *
                                  pad.polarization_factors(beam=beam))
resolution = pad.max_resolution(beam=beam)
mol_size = max_pair_distance(mol.coordinates)
qmax = 2 * np.pi / resolution
mesh_size = int(np.ceil(10 * mol_size / resolution))
mask = pad.beamstop_mask(beam=beam, q_min=4 * np.pi / mol_size).astype(real_t)

a_map_dev = clcore.to_device(shape=(mesh_size**3,), dtype=clcore.complex_t)
q_dev = clcore.to_device(q_vecs, dtype=clcore.real_t)
a_out_dev = clcore.to_device(dtype=clcore.complex_t, shape=pad.shape())
clcore.phase_factor_mesh(mol.coordinates, f, N=mesh_size, q_min=-qmax, q_max=qmax, a=a_map_dev)

n_patterns = 1000
n_orientations = 1
skip = 0

corner = np.array([-qmax]*3, dtype=real_t)
deltas = np.array([((qmax - -qmax) / (mesh_size - 1))] * 3, dtype=real_t)
model = np.zeros(shape=[mesh_size] * 3, dtype=real_t)
weights = np.zeros(shape=[mesh_size] * 3, dtype=real_t)
vec0 = q_vecs.astype(real_t).copy()

live_update = False
if live_update:
    shotim = pg.image(title='Single shot')
    modelim = pg.image(title='Model update')

# outer loop over model updates to be added

for pat in range(n_patterns):
    true_rot = random_rotation().astype(real_t)
    intensity = sim.generate_pattern(rotation=true_rot, poisson=True).astype(real_t)*mask
    if live_update:
        shotim.setImage(intensity)
        w = np.where(weights.flat > 0)
        if len(w[0]) == 1:
            dmodel = np.zeros_like(model)
            dmodel.flat[w] = model.flat[w]/weights.flat[w]
            modelim.setImage(dmodel[int(np.floor(mesh_size[0]/2)), :, :])
        pg.QtGui.QApplication.processEvents()
    w = np.where(intensity.flat > 0)[0]
    intensity_nonzero = intensity.flat[w]
    q_vecs_nonzero = q_vecs[w, :]
    prev_prob = -1e100
    prev_rot = 0
    for orient in range(n_orientations):
        rot = true_rot #random_rotation().astype(real_t)
        model_slice = sim.generate_pattern(rotation=rot, poisson=False).astype(real_t)
        model_slice_nonzero = model_slice.flat[w]
        prob = np.sum(-model_slice_nonzero+intensity_nonzero*log(model_slice_nonzero)-log(factorial(intensity_nonzero)))
        a = np.random.rand(1)
        if a < exp(prob - prev_prob):
            accept = True
        else:
            accept = False
        if orient < skip:
            prev_prob = prob
            prev_rot = rot.copy()
            continue
        if accept is True:
            print('Pattern %d, orientation %d (accept)' % (pat, orient))
            rotq = rotate(rot.T, q_vecs_nonzero)
            trilinear_insertion(model, weights, rotq, intensity_nonzero, corner, deltas)
            prev_prob = prob
            prev_rot = rot.copy()
        else:
            # print('Pattern %d, orientation %d' % (pat, orient))
            rotq = rotate(prev_rot, q_vecs_nonzero)
            trilinear_insertion(model, weights, rotq, intensity_nonzero, corner, deltas)


if 1:
    w = np.where(weights.flat > 0)
    dmodel = np.zeros_like(model)
    dmodel.flat[w] = model.flat[w]/weights.flat[w]
    pg.image(np.log(intensity*mask + 1), title='Single shot')
    pg.image(np.log(dmodel+1), title='Final model')
    pg.image(weights, title='Weights')
    pg.QtGui.QApplication.exec_()
import sys
from time import time
import numpy as np
from numpy import exp, log
from scipy.special import factorial, loggamma, gammaln
from scipy.spatial.transform import Rotation
from scipy import constants
import pyqtgraph as pg
from reborn.detector import PADGeometry
from reborn.source import Beam
from reborn.target.molecule import Molecule
from reborn.target.density import trilinear_interpolation, trilinear_insertion
from reborn.simulate.clcore import ClCore
from reborn.simulate.examples import lysozyme_pdb_file, MoleculeSimulatorV1


r_e = constants.value('classical electron radius')

visualize = False
live_update = 1000
n_patterns = 100
n_orientations = 100
skip = 5
n_model_updates = 2
wavelength = 3e-10
pulse_energy = 10e-3
beam_diameter = 100e-9
n_pixels = 100
pixel_size = 500e-6
distance = 0.05
cl_group_size = 32
rot_angle = (2*np.pi)*1.0

debug_true_rotations = True
debug_correct_starting_model = True

clcore = ClCore(group_size=cl_group_size, double_precision=False)
real_t = clcore.real_t
beam = Beam(wavelength=wavelength, pulse_energy=pulse_energy, diameter_fwhm=beam_diameter)
pad = PADGeometry(n_pixels=n_pixels, pixel_size=pixel_size, distance=distance)
mol = Molecule(pdb_file=lysozyme_pdb_file)
sim = MoleculeSimulatorV1(clcore=clcore, beam=beam, pad=pad, molecule=mol)

q_vecs = pad.q_vecs(beam=beam)
f = mol.get_scattering_factors(beam=beam)
intensity_prefactor = pad.reshape(beam.photon_number_fluence * r_e**2 * pad.solid_angles() *
                                  pad.polarization_factors(beam=beam))
resolution = pad.max_resolution(beam=beam)
mol_size = mol.max_atomic_pair_distance
qmax = 2 * np.pi / resolution
mesh_size = int(np.ceil(6 * mol_size / resolution))
shape = np.array([mesh_size]*3)
mask = pad.beamstop_mask(beam=beam, q_min=2 * np.pi / mol_size).astype(real_t)
w = np.where(mask.flat != 0)
a_map_dev = clcore.to_device(shape=(mesh_size**3,), dtype=clcore.complex_t)
q_dev = clcore.to_device(q_vecs, dtype=clcore.real_t)
a_out_dev = clcore.to_device(dtype=clcore.complex_t, shape=pad.shape())
clcore.phase_factor_mesh(mol.coordinates, f, N=mesh_size, q_min=-qmax, q_max=qmax, a=a_map_dev)
true_model = (np.abs(a_map_dev.get())**2).astype(real_t).reshape([mesh_size]*3)

corner = np.array([-qmax]*3, dtype=real_t)
deltas = np.array([((qmax - -qmax) / (mesh_size - 1))] * 3, dtype=real_t)
weights = np.ones(shape=[mesh_size] * 3, dtype=real_t)
vec0 = q_vecs.astype(real_t).copy()
new_model = np.random.random([mesh_size]*3).astype(real_t)
current_model = new_model.copy()
intensity = sim.generate_pattern(rotation=None, poisson=True).astype(real_t)*mask
new_model *= np.mean(intensity.flat[w])
if debug_correct_starting_model:
    new_model = true_model.copy()

lngam = np.arange(0, 1000)
lngam = loggamma(lngam+1)

patterns = []
rotations = []
qvecs = []

print('='*79)
print('Diffraction detector size: %d x %d' % (n_pixels, n_pixels))
print('Diffraction GPU lookup table size: %d x %d x %d' % (mesh_size, mesh_size, mesh_size))
print('Model grid size: %d x %d x %d' % (mesh_size, mesh_size, mesh_size))
print('='*79)


print('='*79)
sys.stdout.write('Generating patterns... ')
t = time()
pmsg = ''
for pat in range(n_patterns):
    msg = '%d' % pat
    sys.stdout.write('\b'*len(pmsg) + msg)
    pmsg = msg
    true_rot = Rotation.random().as_matrix().astype(real_t)
    rotations.append(true_rot.copy())
    patterns.append(sim.generate_pattern(rotation=true_rot, poisson=True).astype(real_t))
sys.stdout.write('   (%g seconds)' % (time() - t))
print('\n' + ('='*79))

model_slice = pad.zeros().astype(real_t)
prev_prob = 0
prev_rot = 0

##############################
# Generate patterns
###############################

if live_update:
    shotim = pg.image(np.zeros(pad.shape()), title='Single shot')
    shotim.setPredefinedGradient('flame')
    modelim = pg.image(np.zeros(2*[mesh_size]), title='Current model')
    modelim.setPredefinedGradient('flame')
    weightsim = pg.image(np.zeros(2*[mesh_size]), title='Weights')
    weightsim.setPredefinedGradient('flame')
trueim = pg.image(true_model, title='Correct model')
trueim.setPredefinedGradient('flame')

#############################
# Main loop
#############################

current_model_gpu = clcore.to_device(current_model, dtype=real_t)
new_model_gpu = clcore.to_device(new_model, dtype=real_t)
weights_gpu = clcore.to_device(weights, dtype=real_t)
q_vecs_gpu = clcore.to_device(q_vecs, dtype=real_t)

for update in range(n_model_updates):
    current_model_gpu[:] = new_model_gpu
    clcore.divide_nonzero_inplace(current_model_gpu, weights_gpu)
    current_model_gpu.set(current_model)
    weights_gpu.fill(0)
    new_model_gpu.fill(0)
    rot = Rotation.random().as_matrix().astype(real_t)
    for pat in range(n_patterns):
        t = time()
        sys.stdout.write('model %d, pattern %d: ' % (update+1, pat+1))
        true_rot = rotations[pat]
        intensity_gpu = clcore.to_device(patterns[pat], dtype=real_t)
        n_acceptances = 0
        for orient in range(n_orientations):
            rot = true_rot.copy()
            # rot2 = rotation_about_axis(np.random.rand(1)[0]*rot_angle, random_unit_vector()).astype(real_t)
            # rot = rotate(rot2, rot)
            # rotq = rotate(rot.T, q_vecs)
            # rotq_gpu = clcore.to_device(rotq, dtype=real_t)
            # trilinear_interpolation(current_model, rotq, corner, deltas, out=model_slice)
            # model_slice = sim.generate_pattern(rotation=rot, poisson=False).astype(real_t)
            # w = np.where((model_slice > 0)*(intensity > 0)*(mask > 0))[0]
            # M = model_slice.flat[w]
            # K = intensity.flat[w]
            # prob = np.sum(-M + K*log(M) - loggamma(K+1))
            prob = 1
            if orient == 0:
                prev_prob = prob
                prev_rot = rot.copy()
                continue
            # a = np.random.rand(1)
            # prob_ratio = min(exp(np.longdouble(prob) - np.longdouble(prev_prob)), 1)
            # accept = a < prob_ratio
            accept = True
            if accept:
                if orient >= skip:
                    n_acceptances += 1
                    clcore.mesh_insertion(new_model_gpu, weights_gpu, q_vecs_gpu, intensity_gpu, shape=shape,
                                          corner=corner, deltas=deltas, rot=rot)
                prev_prob = prob
                prev_rot = rot.copy()
            else:
                if orient >= skip:
                    clcore.mesh_insertion(new_model_gpu, weights_gpu, q_vecs_gpu, intensity_gpu, shape=shape,
                                          corner=corner, deltas=deltas, rot=rot)
        sys.stdout.write('%5d (%g seconds)\n' % (n_acceptances, time()-t))

new_model = new_model_gpu.get()
weights = weights_gpu.get()
intensity = intensity_gpu.get()

if visualize:
    w = np.where(weights.flat > 0)
    dmodel = np.zeros_like(new_model)
    dmodel.flat[w] = new_model.flat[w] / weights.flat[w]
    if not live_update:
        pg.image(np.log(intensity*mask + 1), title='Single shot')
        pg.image(np.log(dmodel+1), title='Final model')
        pg.image(weights, title='Weights')
    else:
        modelim.setImage(np.log(dmodel+1))
        modelim.setCurrentIndex(int(np.floor(mesh_size / 2)))
        weightsim.setImage(weights)
        weightsim.setCurrentIndex(int(np.floor(mesh_size / 2)))
    pg.QtGui.QApplication.exec_()

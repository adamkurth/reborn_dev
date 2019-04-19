import sys
from time import time
import numpy as np
from numpy import exp, log
from scipy.special import factorial, loggamma, gammaln
import pyqtgraph as pg
from bornagain.detector import PADGeometry
from bornagain.source import Beam
from bornagain.target.molecule import Molecule
from bornagain.target.density import trilinear_interpolation, trilinear_insertion
from bornagain.simulate.clcore import ClCore
from bornagain.simulate.examples import lysozyme_pdb_file, MoleculeSimulatorV1
from bornagain.utils import rotate, random_rotation, random_unit_vector, max_pair_distance, rotation_about_axis
from bornagain.units import r_e
from image_viewers import ImageViewer2

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

debug_true_rotations = False
debug_correct_starting_model = True

real_t = np.float64
clcore = ClCore(group_size=cl_group_size)
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
    true_rot = random_rotation().astype(real_t)
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

live_update = 10
if live_update:
    shotim = pg.image(np.zeros(pad.shape()), title='Single shot')
    shotim.setPredefinedGradient('flame')
    modelim = pg.image(np.zeros(2*[mesh_size]), title='Current model')
    modelim.setPredefinedGradient('flame')
    weightsim = pg.image(np.zeros(2*[mesh_size]), title='Weights')
    weightsim.setPredefinedGradient('flame')
    imview = ImageViewer2()
trueim = pg.image(true_model, title='Correct model')
trueim.setPredefinedGradient('flame')

#############################
# Main loop
#############################

for update in range(n_model_updates):
    current_model = new_model.copy()
    ww = np.where(weights.flat > 0)
    # if len(ww) >= 1:
    current_model.flat[ww] /= weights.flat[ww]
    if live_update > 0:
        modelim.setImage(log(current_model[int(np.floor(mesh_size / 2)), :, :]+1))
        weightsim.setImage(log(weights[int(np.floor(mesh_size / 2)), :, :] + 1))
    weights *= 0
    new_model *= 0
    rot = random_rotation().astype(real_t)
    for pat in range(n_patterns):
        sys.stdout.write('\nmodel %d, pattern %d: ' % (update+1, pat+1))
        true_rot = rotations[pat] #random_rotation().astype(real_t)
        intensity = patterns[pat] #sim.generate_pattern(rotation=true_rot, poisson=True).astype(real_t)*mask
        if live_update > 0 and (pat % live_update) == 0:
            shotim.setImage(log(intensity+1))
            pg.QtGui.QApplication.processEvents()
            imview.set_image(log(intensity+1))
        for orient in range(n_orientations):
            if debug_true_rotations:
                rot = true_rot.copy()
            else:
                rot2 = rotation_about_axis(np.random.rand(1)[0]*rot_angle, random_unit_vector()).astype(real_t)
                rot = rotate(rot2, rot)
            rotq = rotate(rot.T, q_vecs)
            trilinear_interpolation(current_model, rotq, corner, deltas, out=model_slice)
            # model_slice = sim.generate_pattern(rotation=rot, poisson=False).astype(real_t)
            w = np.where((model_slice > 0)*(intensity > 0)*(mask > 0))[0]
            M = model_slice.flat[w]
            K = intensity.flat[w]
            prob = np.sum(-M + K*log(M) - loggamma(K+1))
            if np.isnan(prob).any():
                breakit
            print(prob, np.max(K))
            if orient == 0:
                prev_prob = prob
                prev_rot = rot.copy()
                continue
            a = np.random.rand(1)
            prob_ratio = min(exp(np.longdouble(prob) - np.longdouble(prev_prob)), 1)
            if a < prob_ratio:
                if orient >= skip:
                    sys.stdout.write('%4d ' % (orient+1,))
                    trilinear_insertion(new_model, weights, rotq, intensity, corner, deltas)
                prev_prob = prob
                prev_rot = rot.copy()
            else:
                if orient >= skip:
                    rotq = rotate(prev_rot.T, q_vecs)
                    trilinear_insertion(new_model, weights, rotq, intensity, corner, deltas)


if 1:
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

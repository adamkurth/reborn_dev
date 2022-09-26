import numpy as np
from reborn import source, detector
from reborn.target import placer
from reborn.viewers.qtviews import PADView
from reborn.simulate.clcore import ClCore
drop_diameter = 100e-9
atom_diameter = 1e-9
n_atoms = 10
n_shots = 1000
print(f"{n_atoms} atoms, {n_shots} shots")
beam = source.Beam(photon_energy=7000 * 1.609e-19)
geom = detector.PADGeometry(shape=(101, 101), distance=0.1, pixel_size=100e-6)
clcore = ClCore(double_precision=False, group_size=32)
q_vecs_gpu = clcore.to_device(geom.q_vecs(beam=beam))
amps_gpu = clcore.to_device(shape=geom.n_pixels, dtype=clcore.complex_t)*0
int_gpu = clcore.to_device(shape=geom.n_pixels, dtype=clcore.real_t)*0
for i in range(n_shots):
    print(f"shot {i}")
    print('placing...')
    r_vecs = placer.particles_in_a_sphere(sphere_diameter=drop_diameter, n_particles=n_atoms,
                                          particle_diameter=atom_diameter)
    print('simulating...')
    clcore.phase_factor_qrf(q_vecs_gpu, r_vecs, a=amps_gpu, add=True)
    clcore.mod_squared_complex_to_real(amps_gpu, int_gpu, add=True)
I = np.abs(amps_gpu.get())**2/n_shots
print(f"median value / N = {np.median(I)}, max value / N^2 = {np.max(I)}")
pv = PADView(pad_geometry=geom, beam=beam, data=I)
pv.start()

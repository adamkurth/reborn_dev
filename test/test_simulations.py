import numpy as np
try:
    from bornagain.simulate.clcore import ClCore
except ImportError:
    ClCore = None
import bornagain as ba
from bornagain import detector
from bornagain import source
from bornagain.simulate.examples import lysozyme_pdb_file
from bornagain.target import crystal, density
from scipy import constants as const

hc = const.h*const.c

def test_mappings():

    # There are many ways to simulate crystal patterns.  Here we chack that we get the same results for the molecular
    # transform in the crystal basis and in the "lab" basis.  We also check simulations based on interpolation from
    # a 3D lookup table.

    # Load up the pdb file for PSI
    cryst = crystal.CrystalStructure(lysozyme_pdb_file)
    spacegroup = cryst.spacegroup
    unitcell = cryst.unitcell

    # Coordinates of asymmetric unit in crystal basis x
    au_x_vecs = cryst.fractional_coordinates

    # Setup beam and detector
    beam = source.Beam(wavelength=3e-10)
    pad = detector.PADGeometry(pixel_size=300e-6, distance=0.2, n_pixels=20)
    q_vecs = pad.q_vecs(beam=beam)
    h_vecs = unitcell.q2h(q_vecs)/2/np.pi  # These are the scattering vectors in reciprocal lattice basis

    # Atomic scattering factors (just ones for now -- problems with mapping complex numbers)
    f = 1 + 0 * cryst.molecule.get_scattering_factors(hc / beam.wavelength)

    # Initialize opencl core
    clcore = ClCore()

    # Pass data buffers to GPU device
    amps_mol_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
    amps_mol_gpu2 = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
    amps_slice_gpu = clcore.to_device(shape=pad.shape(), dtype=clcore.complex_t)*0
    q_vecs_gpu = clcore.to_device(q_vecs, dtype=clcore.real_t)
    h_vecs_gpu = clcore.to_device(h_vecs, dtype=clcore.real_t)*2*np.pi
    au_x_vecs_gpu = clcore.to_device(au_x_vecs, dtype=clcore.real_t)
    au_f_gpu = clcore.to_device(f, dtype=clcore.complex_t)

    resolution = 0.1*2*np.pi/np.max(pad.q_mags(beam=beam))
    oversampling = 20
    dens = density.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=oversampling)
    dens_h = dens.h_density_map
    mesh_h_lims = dens_h.limits*2*np.pi
    a_map_dev = clcore.to_device(shape=dens_h.shape, dtype=clcore.complex_t)
    clcore.phase_factor_mesh(au_x_vecs_gpu, au_f_gpu, N=dens_h.shape, q_min=mesh_h_lims[:, 0], q_max=mesh_h_lims[:, 1],
                             a=a_map_dev)

    for i in range(spacegroup.n_molecules):
        mol_x_vecs = spacegroup.apply_symmetry_operation(i, au_x_vecs)
        clcore.phase_factor_qrf(h_vecs_gpu, mol_x_vecs, au_f_gpu, a=amps_mol_gpu, add=True)
        clcore.phase_factor_qrf(q_vecs_gpu, unitcell.x2r(mol_x_vecs), au_f_gpu, a=amps_mol_gpu2, add=True)
        rot = spacegroup.sym_rotations[i]
        trans = spacegroup.sym_translations[i]
        clcore.mesh_interpolation(a_map_dev, h_vecs_gpu, N=dens_h.shape, q_min=mesh_h_lims[:, 0],
                                  q_max=mesh_h_lims[:, 1], a=amps_slice_gpu, R=rot, U=trans, add=True)

    intensities1 = pad.reshape(np.abs(amps_slice_gpu.get())**2)  # From 3D mesh
    intensities2 = pad.reshape(np.abs(amps_mol_gpu.get())**2)    # Atomistic using h, x vectors
    intensities3 = pad.reshape(np.abs(amps_mol_gpu2.get())**2)   # Atomistic using q, r vectors
    # import pyqtgraph as pg
    # pg.image(np.concatenate([intensities1, intensities2, intensities3]))
    # pg.QtGui.QApplication.exec_()
    assert np.mean(np.abs(intensities1 - intensities2))/np.mean(np.abs(intensities2)) < 1e-2  # 1% error tolerance
    # assert np.mean(np.abs(intensities1 - intensities3))/np.mean(np.abs(intensities2)) < 1e-1
    assert np.mean(np.abs(intensities3 - intensities2))/np.mean(np.abs(intensities2)) < 1e-5

import numpy as np
from bornagain.simulate.clcore import ClCore
from bornagain import detector
from bornagain import source
from bornagain.simulate.examples import lysozyme_pdb_file
from bornagain.target import crystal
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
    pad = detector.PADGeometry(pixel_size=300e-6, distance=0.2, shape=(20, 20))
    q_vecs = pad.q_vecs(beam=beam)
    h_vecs = unitcell.q2h(q_vecs)  # These are the scattering vectors in reciprocal lattice basis

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
    dens = crystal.CrystalDensityMap(cryst=cryst, resolution=resolution, oversampling=oversampling)
    mesh_h_lims = dens.h_limits*2*np.pi
    a_map_dev = clcore.to_device(shape=dens.shape, dtype=clcore.complex_t)
    clcore.phase_factor_mesh(au_x_vecs_gpu, au_f_gpu, N=dens.shape, q_min=mesh_h_lims[:, 0], q_max=mesh_h_lims[:, 1],
                             a=a_map_dev)

    for i in range(spacegroup.n_molecules):
        mol_x_vecs = spacegroup.apply_symmetry_operation(i, au_x_vecs)
        clcore.phase_factor_qrf(h_vecs_gpu, mol_x_vecs, au_f_gpu, a=amps_mol_gpu, add=True)
        clcore.phase_factor_qrf(q_vecs_gpu, unitcell.x2r(mol_x_vecs), au_f_gpu, a=amps_mol_gpu2, add=True)
        rot = spacegroup.sym_rotations[i]
        trans = spacegroup.sym_translations[i]
        clcore.mesh_interpolation(a_map_dev, h_vecs_gpu, N=dens.shape, q_min=mesh_h_lims[:, 0],
                                  q_max=mesh_h_lims[:, 1], a=amps_slice_gpu, R=rot, U=trans, add=True)

    intensities1 = pad.reshape(np.abs(amps_slice_gpu.get())**2)  # From 3D mesh
    intensities2 = pad.reshape(np.abs(amps_mol_gpu.get())**2)    # Atomistic using h, x vectors
    intensities3 = pad.reshape(np.abs(amps_mol_gpu2.get())**2)   # Atomistic using q, r vectors
    assert np.mean(np.abs(intensities1 - intensities2))/np.mean(np.abs(intensities2)) < 1e-2  # 1% error tolerance
    assert np.mean(np.abs(intensities3 - intensities2))/np.mean(np.abs(intensities2)) < 1e-5


def test_density_map_fft_vs_direct_sum():

    # Testing compatibility of simulation tools with FFT operations.

    # The CrystalStructure object has a UnitCell, SpaceGroup, and other information.  The input can be any path to a PDB
    # file or it can be the name of a PDB entry.  The PDB will be fetched from the web if necessary and possible.  Some
    # PDB entries (e.g. 2LYZ, 4ET8) come with bornagain.
    cryst = crystal.CrystalStructure('2LYZ')
    cryst = crystal.CrystalStructure('4ET8')

    # The oversampling ratio:
    osr = 2
    # The desired map resolution, which will be adjusted according to crystal lattice and sampling constraints:
    res = 2e-10
    # The CrystalDensityMap is a helper class that ensures sampling in the crystal basis is configured such that
    # the crystal spacegroup symmetry operations of a density map can be performed strictly through permutation operations.
    # Thus, no interpolations are needed for spacegroup symmetry operations.
    cdmap = crystal.CrystalDensityMap(cryst, res, osr)

    # The ClCore instance manages the GPU simulations.
    simcore = ClCore()

    # Create two atom position vectors, both at the origin.
    x_vecs = np.zeros([2, 3])
    # Now shift one of them along the "z" coordinate (in crystal basis) by n steps.  The step size comes from the
    # CrystalDensityMap, which, again, considers how to intelligently sample crystallographic density maps.
    n = 2  # np.round(1/cdmap.dx[2]).astype(int) - 1
    x_vecs[1, 2] = n * cdmap.dx[2]

    # Get some scattering factors
    f = np.array([2, 1j]) #atoms.get_scattering_factors(atomic_numbers=[6, 8], photon_energy=1e4*constants.eV)

    ###############################################
    # METHOD 1:
    ###############################################
    # Simulate amplitudes using atomistic coordinates, structure factors, and a direct summation over
    #                              F(h) =  sum_n f_n*exp(-i 2*pi*h.x_n)
    # Recipcorcal-space coordinates are chosen such that they will correspond to a numpy FFT operation.  The limits of
    # that sample grid are provided by the CrystalDensityMap class:
    g_min = cdmap.h_min * 2 * np.pi
    g_max = cdmap.h_max * 2 * np.pi
    # Simulation tool for regular 3D grid of reciprocal-space samples.
    amps1 = simcore.phase_factor_mesh(x_vecs, f=f, q_min=g_min, q_max=g_max, N=cdmap.shape)
    # Because the phase_factor_mesh function above computes on a grid, the direct 000 voxel is centered.  We must shift
    # the array such that the h=000 is located at the first voxel as per the standard FFT arrangement in numpy.
    amps1 = np.fft.ifftshift(amps1.reshape(cdmap.shape))
    # Transforming from amplitudes to density is now a simple inverse FFT.
    dmap1 = np.fft.ifftn(amps1)

    ##################################################
    # METHOD 2:
    #################################################
    # First make the scattering density map, and then FFT the map to create amplitudes.
    dmap2 = np.zeros(cdmap.shape, dtype=np.complex)
    # Instead of defining a list of atomic coordinates, we directly set the scattering densities to the scattering factors
    # used in METHOD 1.  Note that we've chosen atomic coordinates so that they will lie exactly on grid points in our 3D
    # maps.
    dmap2[0, 0, 0] = f[0]
    dmap2[0, 0, n] = f[1]
    amps2 = np.fft.fftn(dmap2)

    def compare(a, b):
        return np.max(a - b) / np.mean((a + b) / 2)

    assert compare(np.abs(amps1), np.abs(amps2)) < 1e-4
    assert compare(np.abs(np.real(amps1)), np.abs(np.real(amps2))) < 1e-4
    assert compare(np.abs(np.imag(amps1)), np.abs(np.imag(amps2))) < 1e-4


def test_density_map_fft_vs_direct_sum_trilinear():

    # Testing compatibility of simulation tools with FFT operations

    # The CrystalStructure object has a UnitCell, SpaceGroup, and other information.  The input can be any path to a PDB
    # file or it can be the name of a PDB entry.  The PDB will be fetched from the web if necessary and possible.  Some
    # PDB entries (e.g. 2LYZ, 4ET8) come with bornagain.
    cryst = crystal.CrystalStructure('4ET8')

    # The oversampling ratio:
    osr = 2
    # The desired map resolution, which will be adjusted according to crystal lattice and sampling constraints:
    res = 2e-10
    # The CrystalDensityMap is a helper class that ensures sampling in the crystal basis is configured such that
    # the crystal spacegroup symmetry operations of a density map can be performed strictly through permutation operations.
    # Thus, no interpolations are needed for spacegroup symmetry operations.
    cdmap = crystal.CrystalDensityMap(cryst, res, osr)

    # The ClCore instance manages the GPU simulations.
    simcore = ClCore()

    # Create two atom position vectors, both at the origin.
    x_vecs = np.zeros([2, 3])
    # Now shift one of them along the "z" coordinate (in crystal basis) by n steps.  The step size comes from the
    # CrystalDensityMap, which, again, considers how to intelligently sample crystallographic density maps.
    n = 2  # np.round(1/cdmap.dx[2]).astype(int) - 1
    x_vecs[1, 2] = n * cdmap.dx[2]

    # Get some scattering factors
    f = np.array([2, 1]) #atoms.get_scattering_factors(atomic_numbers=[6, 8], photon_energy=1e4*constants.eV)

    ###############################################
    # METHOD 1:
    ###############################################
    # Simulate amplitudes using atomistic coordinates, structure factors, and a direct summation over
    #                              F(h) =  sum_n f_n*exp(-i 2*pi*h.x_n)
    # Recipcorcal-space coordinates are chosen such that they will correspond to a numpy FFT operation.  The limits of
    # that sample grid are provided by the CrystalDensityMap class:
    g_min = cdmap.h_min * 2 * np.pi
    g_max = cdmap.h_max * 2 * np.pi
    # Simulation tool for regular 3D grid of reciprocal-space samples.
    amps1 = simcore.phase_factor_mesh(x_vecs, f=f, q_min=g_min, q_max=g_max, N=cdmap.shape)
    # Because the phase_factor_mesh function above computes on a grid, the direct 000 voxel is centered.  We must shift
    # the array such that the h=000 is located at the first voxel as per the standard FFT arrangement in numpy.
    amps1 = np.fft.ifftshift(amps1.reshape(cdmap.shape))
    # Transforming from amplitudes to density is now a simple inverse FFT.
    dmap1 = np.fft.ifftn(amps1)

    ##################################################
    # METHOD 2:
    #################################################
    # First make the scattering density map, and then FFT the map to create amplitudes.
    dmap2 = np.zeros(cdmap.shape)
    # Instead of defining a list of atomic coordinates, we directly set the scattering densities to the scattering
    # factors used in METHOD 1.  Note that we've chosen atomic coordinates so that they will lie exactly on grid points
    # in our 3D maps.
    dmap2[0, 0, 0] = f[0]
    dmap2[0, 0, n] = f[1]
    dmap3 = cdmap.place_atoms_in_map(x_vecs, f, mode='trilinear')
    assert np.sum(np.abs(dmap2 - dmap3)) < 1e-10
    amps2 = np.fft.fftn(dmap3)

    def compare(a, b):
        return np.max(a - b) / np.mean((a + b) / 2)

    assert compare(np.abs(amps1), np.abs(amps2)) < 1e-4
    assert compare(np.abs(np.real(amps1)), np.abs(np.real(amps2))) < 1e-4
    assert compare(np.abs(np.imag(amps1)), np.abs(np.imag(amps2))) < 1e-4

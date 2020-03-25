import numpy as np
from reborn.target import crystal
from reborn.simulate.examples import lysozyme_pdb_file, psi_pdb_file


def test_read_pdb():

    dic = crystal.pdb_to_dict(psi_pdb_file)
    assert dic is not None

    dic = crystal.pdb_to_dict(lysozyme_pdb_file)
    assert dic is not None


def test_crystal_structure():

    # Check that we can read files and also check that raw SCALE record is correctly generated from the CRYST1 record
    cryst = crystal.CrystalStructure(psi_pdb_file)
    assert np.max(np.abs(cryst.pdb_dict['scale_matrix']*1e10 - cryst.unitcell.o_mat_inv))/np.max(np.abs(cryst.unitcell.o_mat_inv)) < 0.001

    cryst = crystal.CrystalStructure(lysozyme_pdb_file)
    assert np.max(np.abs(cryst.pdb_dict['scale_matrix']*1e10 - cryst.unitcell.o_mat_inv))/np.max(np.abs(cryst.unitcell.o_mat_inv)) < 0.001


def test_spacegroup():

    # Round-trip forward/reverse symmetry operations
    cryst = crystal.CrystalStructure(psi_pdb_file)
    x_vecs = cryst.fractional_coordinates
    x_vecs_mod = cryst.spacegroup.apply_symmetry_operation(3, x_vecs)
    x_vecs_mod = cryst.spacegroup.apply_inverse_symmetry_operation(3, x_vecs_mod)
    assert np.max(np.sqrt(np.sum((x_vecs - x_vecs_mod)**2))) < 1e-6


def test_unitcell():

    # Test some known lattice vectors
    cryst = crystal.CrystalStructure(psi_pdb_file)
    cell = cryst.unitcell
    assert(np.max(np.abs(cell.a_vec - np.array([2.81e-08, 0.00e+00, 0.00e+00]))) < 1e-8)
    assert(np.max(np.abs(cell.a_mat_inv[0, :] - np.array([2.81e-08, 0.00e+00, 0.00e+00]))) < 1e-8)


def test_finite_lattice():

    # Very basic test
    siz = 5
    unitcell = crystal.UnitCell(5e-10, 5e-10, 5e-10, 90*np.pi/180, 90*np.pi/180, 90*np.pi/180)
    lat = crystal.FiniteLattice(max_size=siz, unitcell=unitcell)
    assert(lat.all_x_coordinates.shape[0] == siz**3)
    lat.add_facet(plane=[1, 0, 0], length=2)


def test_density_map_1():

    # Testing h_vecs and that it is consistent with numpy FFT.

    # Load a pdb file
    cryst = crystal.CrystalStructure(lysozyme_pdb_file)

    # Make a densityMap object
    cdmap = crystal.CrystalDensityMap(cryst, resolution=30e-10, oversampling=2)

    # Make a density map and populate two of the voxels, rest is all zeros
    num_atoms = len(cryst.x_vecs)
    # rho = cdmap.place_atoms_in_map(cryst.x_vecs % cdmap.oversampling, np.zeros(num_atoms), mode='trilinear')
    rho = np.zeros(cdmap.shape)
    rho[3, 3, 3] = 1
    rho[5, 6, 7] = 1

    # Define a DFT function
    def dftn(f):
        q_vec = 2 * np.pi * cdmap.h_vecs
        r_vec = cdmap.x_vecs
        shp = f.shape
        f = np.ravel(f)
        ft = np.zeros(f.size, dtype=np.complex64)
        
        for k in range(f.size):
            s = 0
            for n in range(f.size):
                s += f[n] * np.exp(1j * np.dot(q_vec[k, :], r_vec[n, :]))
            ft[k] = s

        return ft.reshape(shp)

    # Calculate the intensity via the numpy fft
    i_fft = np.abs(np.fft.fftn(rho))**2

    # Calculate the intensity via the dft function above
    i_dft = np.abs(dftn(rho))**2
    i_dft = np.fft.ifftshift(i_dft)

    # Calculate the relative error and assert that it be less than some small value
    assert np.sqrt(np.sum((i_fft - i_dft)**2) / np.sum(i_fft**2)) < 1e-6


def test_density_map_2():

    # Round-trip test on au_to_k and k_to_au
    cryst = crystal.CrystalStructure('4ET8', tight_packing=True)
    cdmap = crystal.CrystalDensityMap(cryst, 10e-11, 2)
    dat = np.arange(np.product(cdmap.shape)).reshape(cdmap.shape)
    dat1 = cdmap.au_to_k(0, dat)
    dat2 = cdmap.k_to_au(0, dat1)
    assert np.sum(np.abs(dat - dat1)) != 0
    assert np.sum(np.abs(dat - dat2)) == 0

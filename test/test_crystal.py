# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from reborn.target import crystal
from reborn.data import lysozyme_pdb_file, psi_pdb_file


def test_01():

    dic = crystal.pdb_to_dict(psi_pdb_file)
    assert dic is not None

    dic = crystal.pdb_to_dict(lysozyme_pdb_file)
    assert dic is not None


def test_02():

    # Check that we can read files and also check that raw SCALE record is correctly generated from the CRYST1 record
    cryst = crystal.CrystalStructure(psi_pdb_file)
    assert np.max(np.abs(cryst.pdb_dict['scale_matrix']*1e10 - cryst.unitcell.o_mat_inv))/np.max(np.abs(cryst.unitcell.o_mat_inv)) < 0.001

    cryst = crystal.CrystalStructure(lysozyme_pdb_file)
    assert np.max(np.abs(cryst.pdb_dict['scale_matrix']*1e10 - cryst.unitcell.o_mat_inv))/np.max(np.abs(cryst.unitcell.o_mat_inv)) < 0.001


def test_03():

    # Round-trip forward/reverse symmetry operations
    cryst = crystal.CrystalStructure(psi_pdb_file)
    x_vecs = cryst.fractional_coordinates
    x_vecs_mod = cryst.spacegroup.apply_symmetry_operation(3, x_vecs)
    x_vecs_mod = cryst.spacegroup.apply_inverse_symmetry_operation(3, x_vecs_mod)
    assert np.max(np.sqrt(np.sum((x_vecs - x_vecs_mod)**2))) < 1e-6


def test_04():

    # Test some known lattice vectors
    cryst = crystal.CrystalStructure(psi_pdb_file)
    cell = cryst.unitcell
    assert(np.max(np.abs(cell.a_vec - np.array([2.81e-08, 0.00e+00, 0.00e+00]))) < 1e-8)
    assert(np.max(np.abs(cell.a_mat_inv[0, :] - np.array([2.81e-08, 0.00e+00, 0.00e+00]))) < 1e-8)


def test_05():

    # Very basic test
    siz = 5
    unitcell = crystal.UnitCell(5e-10, 5e-10, 5e-10, 90*np.pi/180, 90*np.pi/180, 90*np.pi/180)
    lat = crystal.FiniteLattice(max_size=siz, unitcell=unitcell)
    assert(lat.all_x_coordinates.shape[0] == siz**3)
    lat.add_facet(plane=[1, 0, 0], length=2)


def test_06():

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


def test_07():

    # Round-trip test on au_to_k and k_to_au
    cryst = crystal.CrystalStructure('4ET8', tight_packing=True)
    cdmap = crystal.CrystalDensityMap(cryst, 10e-11, 2)
    dat = np.arange(np.product(cdmap.shape)).reshape(cdmap.shape)
    dat1 = cdmap.au_to_k(0, dat)
    dat2 = cdmap.k_to_au(0, dat1)
    # assert np.sum(np.abs(dat - dat1)) != 0
    assert np.sum(np.abs(dat - dat2)) == 0


def test_08():

    # Check that au_to_k is the same as symmetry_transform
    for pdb in ['2LYZ', '4ET8']:
        cryst = crystal.CrystalStructure(pdb, tight_packing=True)
        cdmap = crystal.CrystalDensityMap(cryst, 10e-11, 2)
        dat = np.arange(np.product(cdmap.shape)).reshape(cdmap.shape)
        for k in range(cryst.spacegroup.n_molecules):
            dat1 = cdmap.au_to_k(k, dat)
            dat2 = cdmap.symmetry_transform(0, k, dat)
            assert np.sum(np.abs(dat2 - dat1)) == 0


def test_09():

    # Make sure that the bio assemblies are generated properly
    # Tests to see that the number of the atoms divided by the number of bio assemblies
    # should equal the number of atomic symbols in the original pdb.
    for pdb in ['2W0O', '4BED', '2LYZ']:
        cryst = crystal.CrystalStructure(pdb)
        assert int(cryst.fractional_coordinates.shape[0]/cryst.n_bio_partners) == len(cryst.pdb_dict['atomic_symbols'])

        cryst = crystal.CrystalStructure(pdb, create_bio_assembly=0)
        assert int(cryst.fractional_coordinates.shape[0]/cryst.n_bio_partners) == len(cryst.pdb_dict['atomic_symbols'])

        cryst = crystal.CrystalStructure(pdb, create_bio_assembly=1)
        assert int(cryst.fractional_coordinates.shape[0]/cryst.n_bio_partners) == len(cryst.pdb_dict['atomic_symbols'])

    for pdb in ['1SS8']:
        cryst = crystal.CrystalStructure(pdb)
        assert int(cryst.fractional_coordinates.shape[0]/cryst.n_bio_partners) == len(cryst.pdb_dict['atomic_symbols'])

        cryst = crystal.CrystalStructure(pdb, create_bio_assembly=0)
        assert int(cryst.fractional_coordinates.shape[0]/cryst.n_bio_partners) == len(cryst.pdb_dict['atomic_symbols'])

        cryst = crystal.CrystalStructure(pdb, create_bio_assembly=1)
        assert int(cryst.fractional_coordinates.shape[0]/cryst.n_bio_partners) == len(cryst.pdb_dict['atomic_symbols'])

        cryst = crystal.CrystalStructure(pdb, create_bio_assembly=2)
        assert int(cryst.fractional_coordinates.shape[0]/cryst.n_bio_partners) == len(cryst.pdb_dict['atomic_symbols'])




def test_spacegroup():

    hmsymb = 'I 41/a -3 2/d'
    hall_number = crystal.hall_number_from_hermann_mauguin_symbol(hmsymb)
    assert(hall_number == 530)
    itoc_number = crystal.itoc_number_from_hermann_mauguin_symbol(hmsymb)
    assert(itoc_number == 230)
    hmsymb2 = crystal.hermann_mauguin_symbol_from_hall_number(hall_number)
    assert(hmsymb == hmsymb2)
    itoc_number = 1
    hall_number = crystal.hall_number_from_itoc_number(itoc_number)
    assert(hall_number == 1)
    hm_symb = crystal.hermann_mauguin_symbol_from_itoc_number(itoc_number)
    assert(hm_symb == 'P 1')
    itoc = crystal.itoc_number_from_hall_number(1)
    assert(itoc == 1)

    sg = crystal.SpaceGroup(itoc_number=1)
    assert(sg.n_molecules == 1)
    assert(sg.hermann_mauguin_symbol == 'P 1')

    # Check that translations are multiples of 1, 1/2, 1/3, 1/4, or 1/6.
    uniqtrans = []
    reductrans = []
    for h in range(1, 531):
        sg = crystal.SpaceGroup(hall_number=h)
        trans = [[],[],[]]
        transc = [[],[],[]]
        for vec in sg.sym_translations:
            for j in range(0, 3):
                comp = vec[j] % 1
                comp = min(comp, 1-comp)
                if comp == 0:
                    comp = 1
                comp = int(np.round(1/comp))
                if comp not in trans[j]:
                    trans[j].append(comp)
        for j in range(0, 3):
            tr = np.sort(np.array(trans[j], dtype=int))[::-1]
            trans[j] = list(tr)
            indiv = [tr[0]]
            for p in range(0, len(tr)):
                for q in range(0, len(tr)):
                    rat = max(tr[p], tr[q])/float(max(tr[p], tr[q]))
                    if np.abs(rat - np.round(rat)) > 1e-2:
                        if tr[q] not in indiv:
                            indiv.append(tr[p])
            transc[j] = indiv
        uniqtrans.append(trans)
        reductrans.append(transc)

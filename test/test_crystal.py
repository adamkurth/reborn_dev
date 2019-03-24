import sys
sys.path.append('..')

import numpy as np
from bornagain import target
from bornagain.target import crystal


def test_load_pdb_and_assemble():
    # print("\n Entering quick test:")
    pdb_struct = target.crystal.CrystalStructure("../examples/data/pdb/2LYZ.pdb")
    lat_vecs = target.crystal.assemble(pdb_struct.unitcell.o_mat, 10)

    # print ("\tMade cubic lattice with bounds %.2f-%.2f, %.2f-%.2f, %.2f-%.2f Angstrom" %\
    # tuple( np.ravel( [ (i,j) for i,j in zip( lat_vecs.min(0)*1e10, lat_vecs.max(0)*1e10 )]) ))
    lat_vecs_rect = target.crystal.assemble(pdb_struct.unitcell.o_mat, (10, 10, 20))
    # print ("\tMade rectangular lattice with bounds %.2f-%.2f, %.2f-%.2f, %.2f-%.2f Angstrom" %\
    #    tuple( np.ravel( [ (i,j) for i,j in zip( lat_vecs_rect.min(0)*1e10, lat_vecs_rect.max(0)*1e10 )]) ))
    assert(lat_vecs_rect.max(0)[-1] > lat_vecs.max(0)[-1])


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


def test_finite_lattice():

    siz = 5
    unitcell = crystal.UnitCell(5e-10, 5e-10, 5e-10, 90*np.pi/180, 90*np.pi/180, 90*np.pi/180)
    lat = crystal.FiniteLattice(max_size=siz, unitcell=unitcell)
    assert(lat.all_x_coordinates.shape[0] == siz**3)
    lat.add_facet(plane=[1, 0, 0], length=2)


if __name__ == "__main__":
    test_load_pdb_and_assemble()

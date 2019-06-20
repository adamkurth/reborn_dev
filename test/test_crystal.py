import sys
sys.path.append('..')

import numpy as np
from bornagain import target
from bornagain.target import crystal, spgrp
from bornagain.simulate.examples import lysozyme_pdb_file, psi_pdb_file


def test_load_pdb_and_assemble():
    # print("\n Entering quick test:")
    pdb_struct = target.crystal.CrystalStructure(lysozyme_pdb_file)
    lat_vecs = target.crystal.assemble(pdb_struct.unitcell.o_mat, 10)

    # print ("\tMade cubic lattice with bounds %.2f-%.2f, %.2f-%.2f, %.2f-%.2f Angstrom" %\
    # tuple( np.ravel( [ (i,j) for i,j in zip( lat_vecs.min(0)*1e10, lat_vecs.max(0)*1e10 )]) ))
    lat_vecs_rect = target.crystal.assemble(pdb_struct.unitcell.o_mat, (10, 10, 20))
    # print ("\tMade rectangular lattice with bounds %.2f-%.2f, %.2f-%.2f, %.2f-%.2f Angstrom" %\
    #    tuple( np.ravel( [ (i,j) for i,j in zip( lat_vecs_rect.min(0)*1e10, lat_vecs_rect.max(0)*1e10 )]) ))
    assert(lat_vecs_rect.max(0)[-1] > lat_vecs.max(0)[-1])


def test_unitcell():

    cryst = target.crystal.CrystalStructure(psi_pdb_file)
    cell = cryst.unitcell
    assert(np.max(np.abs(cell.a_vec - np.array([2.81e-08, 0.00e+00, 0.00e+00]))) < 1e-8)
    assert(np.max(np.abs(cell.a_mat_inv[0, :] - np.array([2.81e-08, 0.00e+00, 0.00e+00]))) < 1e-8)


def test_spacegroup():

    hmsymb = 'I 41/a -3 2/d'
    # hall_number = crystal.hall_number_from_hermann_mauguin_symbol(hmsymb)
    # assert(hall_number == 530)
    # itoc_number = crystal.itoc_number_from_hermann_mauguin_symbol(hmsymb)
    # assert(itoc_number == 230)
    # hmsymb2 = crystal.hermann_mauguin_symbol_from_hall_number(hall_number)
    # assert(hmsymb == hmsymb2)
    # itoc_number = 1
    # hall_number = crystal.hall_number_from_itoc_number(itoc_number)
    # assert(hall_number == 1)
    # hm_symb = crystal.hermann_mauguin_symbol_from_itoc_number(itoc_number)
    # assert(hm_symb == 'P 1')
    # itoc = crystal.itoc_number_from_hall_number(1)
    # assert(itoc == 1)

    sg = crystal.SpaceGroup(itoc_number=1)
    assert sg.n_molecules == 1
    assert sg.hermann_mauguin_symbol == 'P 1'

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
            tr = np.sort(np.array(trans[j], dtype=np.int))[::-1]
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

    # import spglib
    # a = []
    # print('')
    # for i in range(1, 531):
    #     s = spglib.get_spacegroup_type(i)['international_full']
    #     print(s)
    #     if s in a:
    #         print('---------------------------->', s)
    #     a.append(s)
    # assert len(np.unique(a)) == 530


def test_finite_lattice():

    siz = 5
    unitcell = crystal.UnitCell(5e-10, 5e-10, 5e-10, 90*np.pi/180, 90*np.pi/180, 90*np.pi/180)
    lat = crystal.FiniteLattice(max_size=siz, unitcell=unitcell)
    assert(lat.all_x_coordinates.shape[0] == siz**3)
    lat.add_facet(plane=[1, 0, 0], length=2)


if __name__ == "__main__":
    test_spacegroup()

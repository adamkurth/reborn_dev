import sys
sys.path.append('..')

from bornagain import target

def test_load_pdb_and_assemble():
    # print("\n Entering quick test:")
    pdb_struct = target.crystal.CrystalStructure("../examples/data/pdb/2LYZ.pdb")
    lat_vecs = target.crystal.assemble(pdb_struct.O, 10) 

    #print ("\tMade cubic lattice with bounds %.2f-%.2f, %.2f-%.2f, %.2f-%.2f Angstrom" %\
    #    tuple( np.ravel( [ (i,j) for i,j in zip( lat_vecs.min(0)*1e10, lat_vecs.max(0)*1e10 )]) ))
    lat_vecs_rect = target.crystal.assemble(pdb_struct.O, (10,10,20))
    #print ("\tMade rectangular lattice with bounds %.2f-%.2f, %.2f-%.2f, %.2f-%.2f Angstrom" %\
    #    tuple( np.ravel( [ (i,j) for i,j in zip( lat_vecs_rect.min(0)*1e10, lat_vecs_rect.max(0)*1e10 )]) ))
    assert( lat_vecs_rect.max(0)[-1] > lat_vecs.max(0)[-1] )


if __name__ =="__main__":
    test_load_pdb_and_assemble()



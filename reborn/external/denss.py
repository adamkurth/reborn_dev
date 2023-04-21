import os
import numpy as np
from reborn.target import crystal, atoms
import saxstats.saxstats as saxstats


def create_density_map(
    pdb_file,
    solvent_contrast=True,
    cell=100e-9,
    create_bio_assembly=True,
    map_resolution=0.2e-9,
    map_oversampling=2,
):
    cryst = crystal.CrystalStructure(
        pdb_file,
        spacegroup="P1",
        unitcell=(cell, cell, cell, np.pi/2, np.pi/2, np.pi/2),
        create_bio_assembly=create_bio_assembly,
    )
    cell = cryst.molecule.max_atomic_pair_distance
    cryst = crystal.CrystalStructure(
        pdb_file,
        spacegroup="P1",
        unitcell=(cell, cell, cell, np.pi/2, np.pi/2, np.pi/2),
        create_bio_assembly=create_bio_assembly,
    )
    dmap = crystal.CrystalDensityMap(cryst, map_resolution, map_oversampling)
    pdb = saxstats.PDB(pdb_file)
    n = pdb.natoms
    s = int(cryst.molecule.n_atoms/n)
    assert(cryst.molecule.n_atoms/n % 1 == 0)
    if s > 1:
        pdb.natoms = cryst.molecule.n_atoms
        pdb.atomnum = np.concatenate([pdb.atomnum]*s)
        pdb.atomname = np.concatenate([pdb.atomname]*s)
        pdb.atomalt = np.concatenate([pdb.atomalt]*s)
        pdb.resname = np.concatenate([pdb.resname]*s)
        pdb.resnum = np.concatenate([pdb.resnum]*s)
        pdb.chain = np.concatenate([pdb.chain]*s)  # FIXME: Unique names needed here?
        pdb.coords = cryst.molecule.coordinates*1e10
        pdb.occupancy = np.concatenate([pdb.occupancy]*s)
        pdb.b = np.concatenate([pdb.b]*s)
        pdb.atomtype = np.concatenate([pdb.atomtype]*s)
        pdb.charge = np.concatenate([pdb.charge]*s)
        pdb.nelectrons = np.concatenate([pdb.nelectrons]*s)
        pdb.radius = np.zeros(n*s) #np.concatenate([pdb.radius]*s)
        pdb.vdW = np.concatenate([pdb.vdW]*s)
        pdb.unique_volume = np.zeros(n*s) #np.concatenate([pdb.unique_volume]*s)
        pdb.unique_radius = np.zeros(n*s) #np.concatenate([pdb.unique_radius]*s)
        pdb.numH = np.concatenate([pdb.numH]*s)
    voxel = cell / dmap.cshape[0]
    side = dmap.shape[0] * voxel
    pdb2mrc = saxstats.PDB2MRC(pdb=pdb, voxel=voxel * 1e10, side=side * 1e10)
    pdb2mrc.scale_radii()
    pdb2mrc.make_grids()
    pdb2mrc.calculate_resolution()
    pdb2mrc.calculate_invacuo_density()
    pdb2mrc.calculate_excluded_volume()
    pdb2mrc.calculate_hydration_shell()
    pdb2mrc.calc_rho_with_modified_params(pdb2mrc.params)
    if solvent_contrast:
        print("solvent")
        rho = pdb2mrc.rho_insolvent
    else:
        rho = pdb2mrc.rho_invacuo
    side = pdb2mrc.side
    return rho, side*1e-10

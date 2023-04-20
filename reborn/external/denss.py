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
    if s > 1:
        pdb.atomnum = np.concatentate([pdb.atomnum]*s)
        pdb.atomname = np.concatentate([pdb.atomname]*s)
        pdb.atomalt = np.concatentate([pdb.atomalt]*s)
        pdb.resname = np.concatentate([pdb.resname]*s)
        pdb.resnum = np.concatentate([pdb.resnum]*s)
        pdb.chain = np.concatentate([pdb.chain]*s)
        pdb.coords = np.concatentate([pdb.coords]*s)
        pdb.occupancy = np.concatentate([pdb.occupancy]*s)
        pdb.b = np.concatentate([pdb.b]*s)
        pdb.atomtype = np.concatentate([pdb.atomtype]*s)
        pdb.charge = np.concatentate([pdb.charge]*s)
        pdb.nelectrons = np.concatentate([pdb.nelectrons]*s)
        pdb.radius = np.concatentate([pdb.radius]*s)
        pdb.vdW = np.concatentate([pdb.vdW]*s)
        pdb.unique_volume = np.concatentate([pdb.unique_volume]*s)
        pdb.unique_radius = np.concatentate([pdb.unique_radius]*s)
        #set a variable with H radius to be used for exvol radii optimization
        #set a variable for number of hydrogens bonded to atoms
        # pdb.exvolHradius = radius['H']
        pdb.numH = np.concatentate([pdb.numH]*s)

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
        rho = pdb2mrc.rho_insolvent
    else:
        rho = pdb2mrc.rho_invacuo
    # rho = ifftshift(rho)
    side = pdb2mrc.side
    rho = np.fft.ifftshift(rho.real)
    return rho, side

import pkg_resources

# PDB files
lysozyme_pdb_file = pkg_resources.resource_filename('bornagain', 'data/pdb/2LYZ.pdb')
psi_pdb_file = pkg_resources.resource_filename('bornagain', 'data/pdb/1jb0.pdb')

# CrystFEL geom files
pnccd_geom_file = pkg_resources.resource_filename('bornagain', 'data/geom/pnccd_front.geom')
cspad_geom_file = pkg_resources.resource_filename('bornagain', 'data/geom/cspad.geom')

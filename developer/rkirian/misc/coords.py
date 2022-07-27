from reborn.target import crystal
cryst = crystal.CrystalStructure(pdb_file_path='4BED', expand_ncs_coordinates=False, create_bio_assembly=True)
cryst.molecule.view()

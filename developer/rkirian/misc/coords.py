from reborn.target import crystal
from reborn.viewers.qtviews import qtviews

expand = False

cryst = crystal.CrystalStructure(pdb_file_path='4BED', expand_ncs_coordinates=False, create_bio_assembly=True)
print(cryst)
r = cryst.molecule.coordinates

v = qtviews.Scatter3D()
v.add_points(r)
v.show()

print(cryst)

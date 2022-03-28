from reborn.simulate import engines
from reborn.target import crystal
from reborn.viewers.qtviews import padviews
from reborn import detector, source, const


# define get_data for framegetter like this:

mol = crystal.CrystalStructure(pdb_file_path='2LYZ').molecule
pads = detector.cspad_2x2_pad_geometry_list(detector_distance=0.5)
beam = source.Beam(photon_energy=6000*const.eV, pulse_energy=1e-3, diameter_fwhm=200e-9)

engine = engines.MoleculePatternSimulator(beam=beam, molecule=mol, pad=pads, oversample=8)

for i in range(1000):
    print(i, end='\n')
    intensity = engine.generate_pattern()

padviews.view_pad_data(pad_geometry=pads, pad_data=intensity)


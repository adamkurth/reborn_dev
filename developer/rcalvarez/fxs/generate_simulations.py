from pdb_simulator import Simulator
from reborn.const import eV
from reborn.detector import jungfrau4m_pad_geometry_list
from reborn.source import Beam
from reborn.viewers.qtviews import PADView

pdb_id = '2LYZ'
beam = Beam(photon_energy=9e3*eV)
pads = jungfrau4m_pad_geometry_list(detector_distance=1)
fg = Simulator(pad_geometry=pads, beam=beam, pdb=pdb_id)
pv = PADView(frame_getter=fg)
pv.start()

import h5py
from pdb_simulator import Simulator
from reborn.const import eV
from reborn.detector import jungfrau4m_pad_geometry_list
from reborn.source import Beam
from reborn.viewers.qtviews import PADView

pdb_id = '2LYZ'
photon_energy = 9e3 * eV
beam = Beam(photon_energy=photon_energy)
detector_distance = 1
pads = jungfrau4m_pad_geometry_list(detector_distance=detector_distance)
fg = Simulator(pad_geometry=pads, beam=beam, pdb=pdb_id)

for i, f in enumerate(fg):
    with h5py.File(f'./simulations/{pdb_id}/frame_{i:04}.h5', 'w') as hf:
        hf.create_dataset('beam/photon_energy', data=photon_energy)
        hf.create_dataset('geometry/detector_distance', data=detector_distance)

        rawd = f.get_raw_data_flat()
        hf.create_dataset('data', data=rawd)

# pv = PADView(frame_getter=fg)
# pv.start()

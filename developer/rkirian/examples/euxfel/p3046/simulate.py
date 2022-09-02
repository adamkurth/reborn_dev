from reborn import source
from reborn.const import eV
from reborn.simulate import solutions
from reborn.viewers.qtviews.padviews import PADView
from reborn.external import crystfel
geom_file = 'geometry/agipd_september_2022_v03.geom'
geom = crystfel.geometry_file_to_pad_geometry_list(geom_file)
geom.translate([0, 0, 0.1])
beam = source.Beam(photon_energy=8000*eV, pulse_energy=2e-3)
intensity = solutions.get_pad_solution_intensity(pad_geometry=geom, beam=beam, thickness=2.5e-6, poisson=True)
pv = PADView(pad_geometry=geom, data=intensity, beam=beam)
pv.start()

from bornagain.simulate import examples
from bornagain.viewers.qtviews import PADView

sim = examples.lysozyme_molecule()

padgui = PADView(data=sim['intensity'], pad_geometry=sim['pad_geometry'])
padgui.add_rings([100, 200, 300])
padgui.start()

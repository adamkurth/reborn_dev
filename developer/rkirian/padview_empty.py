import sys
sys.path.append('../..')

from bornagain.viewers.qtviews import PADView

padview = PADView()#pad_geometry=pad_geometry, frame_getter=frame_getter, mask_data=masks)
padview.start()
sys.exit()

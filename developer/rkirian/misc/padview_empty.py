import sys
sys.path.append('../..')

from bornagain.viewers.qtviews import PADView

padview = PADView()
padview.start()
sys.exit()

import sys
sys.path.append('../..')

from reborn.viewers.qtviews import PADView

padview = PADView()
padview.start()
sys.exit()

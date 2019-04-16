import sys
sys.path.append('../..')

import getters

streamfile_path = '../../examples/data/crystfel/streamfiles/r0123-KY-sorted_modifiedForTesting.stream'


c = getters.StreamfileFrameGetter(streamfile_path)

print(c.n_frames)
print(c.get_frame(1))
print(c.streamfile_name)
import sys
sys.path.append('../..')

import getters

streamfile_path = '../../examples/data/crystfel/streamfiles/r0123-KY-sorted_modifiedForTesting.stream'


c = getters.StreamfileFrameGetter(streamfile_path)

# print(c.n_frames)
# print(c.streamfile_name)
# print(c.get_frame(1))

a = c.get_frame(4)
print(a['A_matrix'])
print(a['cxiFilepath'])
print(a['cxiFileFrameNumber'])


# Think about zeroth frame, last frame, frame that does not contain crystal
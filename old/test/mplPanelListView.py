import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from bornagain import dataio





im = np.random.random([10,20])

fig, ax = plt.subplots()
ax.set_xlim([0,20])
ax.set_ylim([0,20])


trans = mpl.transforms.Affine2D().rotate_deg(10) # + ax.transData

ax.imshow(im,interpolation='none',transform=trans)

# ax.set_transform(trans)


fig.canvas.draw() 
plt.show()








# 
# 
# 
# [pa, reader] = convert.crystfelToPanelList("examples/example1.geom")
# reader.getShot(pa, "examples/example1.h5")
# 
# bbox = pa.realSpaceBoundingBox
# 
# axim = []
# 
# fig, ax = plt.subplots()
# ax.set_xlim([bbox[0, 0], bbox[1, 0]])
# ax.set_ylim([bbox[0, 1], bbox[1, 1]])
# # ax.invert_xaxis()
# # ax.invert_yaxis()
# 
# for i in np.arange(1):
# 
#     p = pa[i]
#     pbbox = p.realSpaceBoundingBox
#     im = np.zeros([200, 400])  # p.data
# 
#     im[2:20, 0:100] = 1  # np.max(im)
# 
#     print(i)
#     print(bbox)
#     print(pbbox)
#     print(p.T)
#     print(p.F)
#     print(p.S)
# 
# 
#     trans = mpl.transforms.Affine2D().scale(1,1) + ax.transData
# 
#     axim.append(ax.imshow(im, aspect='equal', extent=np.array([0, im.shape[1], 0, im.shape[0]]), 
#                           alpha=1, zorder=-1, interpolation='nearest', cmap='gray', 
#                           origin='lower'))
#                           
#     axim[0].set_transform(trans)
#     plt.draw()
#     
#     
# 
# plt.show()



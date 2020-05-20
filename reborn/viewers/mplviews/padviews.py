import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from reborn.detector import concat_pad_data
from reborn import utils


def view_pad_data(pad_data, pad_geometry, pad_numbers=False, circle_radii=None, show=True):
    r"""
    Very simple function to show pad data with matplotlib.  This will take a list of data arrays along with a list
    of |PADGeometry| instances and display them with a decent geometrical layout.

    Arguments:
        pad_data:
        pad_geometry:
        pad_numbers:
        circle_radii:

    Returns:
        axis
    """
    pads = utils.ensure_list(pad_geometry)
    data = utils.ensure_list(pad_data)
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_facecolor(np.array([0, 0, 0])+0.2) #'dimgray')
    q_max = np.max(concat_pad_data(pad_data))
    q_min = np.min(concat_pad_data(pad_data))
    imshow_args = {"vmin": q_min, "vmax": q_max, "interpolation": 'none', "cmap": 'gnuplot'}
    bbox = []
    for i in range(len(pads)):
        dat = data[i]
        pad = pads[i]
        f = pad.fs_vec.copy()
        s = pad.ss_vec.copy()
        t = pad.t_vec.copy()
        c = t + f * dat.shape[0] / 2 + s * dat.shape[1] / 2
        scl = pad.pixel_size()
        f /= scl
        s /= scl
        t /= scl
        c /= scl
        # This bbox is for finding the bounding box of all panels -- need coords of all four corners of each...
        bbox.append(np.array([[t[0], t[1]], [t[0]+f[0], t[1]+f[1]], [t[0]+s[0], t[1]+s[1]],
                              [t[0]+f[0]+s[0], t[1]+f[1]+s[1]]]))
        im = ax.imshow(dat, **imshow_args)
        trans = mpl.transforms.Affine2D(np.array([[f[0], s[0], t[0]],
                                                  [f[1], s[1], t[1]],
                                                  [   0,    0,    1]])) + ax.transData
        im.set_transform(trans)
        if pad_numbers:
            ax.text(c[0], c[1], s=str(i), color='c', horizontalalignment='center', verticalalignment='center')
    if circle_radii is not None:
        for r in utils.ensure_list(circle_radii):
            # circ = plt.Circle(xy=(0, 0), radius=r) #, fc='none', ec='C2') #, ls='dashed')
            ax.add_patch(plt.Circle(xy=(0, 0), radius=r, fc='none', ec=[0, 1, 0]))
    b = np.max(np.abs(np.vstack(bbox)))+1
    ax.set_xlim(-b, b)  # work in pixel units
    ax.set_ylim(b, -b)
    if show:
        plt.show()

    return ax

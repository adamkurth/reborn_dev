import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from reborn.detector import concat_pad_data
from reborn import utils


def view_pad_data(pad_data, pad_geometry, pad_numbers=False, beam_center=False, show_scans=False, show_coords=False,
                  show=True, vmin=None, vmax=None, background_color=None):
    r"""
    Very simple function to show pad data with matplotlib.  This will take a list of data arrays along with a list
    of |PADGeometry| instances and display them with a decent geometrical layout.

    Arguments:

    Returns:
        axis
    """
    pads = utils.ensure_list(pad_geometry)
    data = utils.ensure_list(pad_data)
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_facecolor(np.array([0, 0, 0])+0.2)
    if background_color is not None:
        ax.set_facecolor(background_color)

    pad_data_concated = concat_pad_data(pad_data)

    if vmin == None:
        vmin = np.min(pad_data_concated)

    if vmax == None:
        vmax = np.max(pad_data_concated)

    imshow_args = {"vmin": vmin, "vmax": vmax, "interpolation": 'none', "cmap": 'viridis'}
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
            ax.text(c[0], c[1], s=str(i), color='c', ha='center', va='center', bbox=dict(boxstyle="square",
                   ec=(0.5, 0.5, 0.5), fc=(0.3, 0.3, 0.3), alpha=0.5
                   ))
        if show_scans:
            plt.arrow(t[0], t[1], f[0]*dat.shape[0]/2, f[1]*dat.shape[1]/2, fc='b', ec='r', width=10,
                      length_includes_head=True)
    b = np.max(np.abs(np.vstack(bbox)))+1
    ax.set_xlim(-b, b)  # work in pixel units
    ax.set_ylim(b, -b)
    if beam_center:
        ax.add_patch(plt.Circle(xy=(0, 0), radius=b/100, fc='none', ec=[0, 1, 0]))
    if show_coords:
        plt.arrow(0, 0, b/10, 0, fc=[1, 0, 0], ec=[1, 0, 0], width=10, length_includes_head=True)
        plt.arrow(0, 0, 0, b / 10, fc=[0, 1, 0], ec=[0, 1, 0], width=10, length_includes_head=True)
        ax.add_patch(plt.Circle(xy=(0, 0), radius=10, fc=[0, 0, 1], ec=[0, 0, 1], zorder=100))
    if show:
        plt.show()

    return ax, im

import numpy as np

from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
from matplotlib.figure import Figure
import matplotlib


def stretch(two, scale):

    ran = (two[1] - two[0])
    cent = (two[1] + two[0]) / 2.0
    return [cent - ran * scale / 2.0, cent + ran * scale / 2.0]


def connectAxisZoomFunction(fig, base_scale=1.5, x_scale=None, y_scale=None):

    """ Connect a zoom function to an axis (zoom when mouse wheel is scrolled)"""

    def zoom_fun(event):

        ax = event.inaxes
        print(ax)

        zoomx = True
        zoomy = True

        xs = base_scale
        ys = base_scale

        if any((x_scale, y_scale)):

            zoomx = False
            zoomy = False

            if x_scale is not None:
                xs = x_scale
                zoomx = True

            if y_scale is not None:
                ys = y_scale
                zoomy = True

        if ax is not None:

            if event.button == 'up':
                xs = 1.0 / xs
                ys = 1.0 / ys
            elif event.button == 'down':
                pass
            else:
                return

            if zoomx is True:
                ax.set_xlim(stretch(ax.get_xlim(), xs))

            if zoomy is True:
                ax.set_ylim(stretch(ax.get_ylim(), ys))

            plt.draw()

    fig.canvas.mpl_connect('scroll_event', zoom_fun)



def setAxesColor(ax, col):
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color(col)
    ax.xaxis.label.set_color(col)
    ax.yaxis.label.set_color(col)
    ax.tick_params(axis='x', colors=col)
    ax.tick_params(axis='y', colors=col)



class mplColorBar(object):

    pass

class mplPanelView(object):

    def __init__(self, im=None):

        # create the figure
        self.fig = plt.figure()
        self.fig.clf()
        self.fig.patch.set_facecolor([0, 0, 0])

        # create the image axes
        self.im_ax = self.fig.add_axes([0, 0, 1, 1])
        self.im_ax.patch.set_facecolor([0, 0, 0])
        self.im_ax.set_axis_off()


        self.img = None
        self.imageRange = None
        if im is not None:
            self.setImage(im)

        # create colorbar view
        self.setup_cbar()

        # view settings
        self.baseCmap = cm.datad['CMRmap']
        self.set_gamma(1)

        # setup zooming feature
        self.zoomScale = 1.3
        connectAxisZoomFunction(self.fig, base_scale=self.zoomScale)

    def set_gamma(self, gamma=None):

        if gamma is not None:
            self.gamma = gamma
        self.cmap = colors.LinearSegmentedColormap('test', self.baseCmap, \
 gamma=self.gamma)
        try:
            self.img.set_cmap(self.cmap)
            self.cb_ax_img.set_cmap(self.cmap)
            self.fig.canvas.draw()
        finally:
            pass


    def setImage(self, im):

        self.image = im
        self.imageRange = [np.min(im), np.max(im)]

        if self.img is None:
            self.img = self.im_ax.imshow(self.image)
            self.img.set_interpolation('nearest')
#            self.img.set_cmap(self.cmap)
            self.fig.show()
        else:
            self.img.set_data(im)

        self.fig.canvas.draw()


    def setup_cbar(self):

        # view range for colorbar/histogram
        histRange = stretch(self.imageRange, 1.05)

        # histogram axes
        self.hs_ax = self.fig.add_axes([0.92, 0.05, 0.06, 0.9])
        self.hs_ax.patch.set_facecolor([0, 0, 0])
        self.hs_ax.patch.set_alpha(0)
        setAxesColor(self.hs_ax, [.5, .5, .5])
        self.hs_ax.set_ylim(histRange)

        # plot histogram
        self.hs_ax_hs = self.hs_ax.hist(self.image.ravel(), bins=200, log=True, \
         normed=True, orientation='horizontal', histtype='stepfilled', \
         color=[.2, .2, .2])
        self.hs_ax.set_xticks([])
        plt.setp(self.hs_ax.get_yticklabels(), fontsize=10)
        cde = stretch([self.imageRange[0], self.imageRange[1]], 1.05)

        # colorbar data
        cdat = np.arange(cde[0], cde[1] + 1e-10, (cde[1] - cde[0]) / 1000)
        cdat = np.tile(cdat, [100, 1])
        cdat = cdat.T
        self.cdat = cdat

        # colorbar axes
        self.cb_ax = self.fig.add_axes([0.98, 0.05, 0.01, 0.9], sharey=self.hs_ax)
        self.cb_ax.patch.set_facecolor([0, 0, 0])
        self.hs_ax.set_xlim([0, 1])
        setAxesColor(self.cb_ax, [.5, .5, .5])
        self.cb_ax.set_xticks([])
        plt.setp(self.cb_ax.get_yticklabels(), visible=False)

        # colorbar image
        self.cb_ax_img = self.cb_ax.imshow(cdat, \
         extent=(-1000, 1000, histRange[0], histRange[1]), \
         origin='lower', interpolation='nearest')
        self.cb_ax_img.set_clim(self.img.get_clim())

        self.fig.canvas.draw()


    def set_clim(self, lim):

        self.img.set_clim(lim)
        self.cb_ax_img.set_clim(lim)
        self.fig.canvas.draw()

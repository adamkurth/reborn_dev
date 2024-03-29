# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

from sphinx_gallery import scrapers
import pyqtgraph as pg
from pyvirtualdisplay import Display

global qt_app
qt_app = pg.mkQApp()

global display
display = None

def qtscraper(block, block_vars, gallery_conf):
    """Basic implementation of a Qt window scraper.

    Looks for any non-hidden windows in the current application instance and
    uses ``grab`` to render an image of the window. The window is closed
    afterward, so you have to call ``show()`` again to render it in a
    subsequent cell.

    ``processEvents`` is called once in case events still need to propagate.

    Intended for use in ``image_scrappers`` in the sphinx-gallery configuration.
    """
    imgpath_iter = block_vars['image_path_iterator']

    global qt_app
    qt_app = pg.mkQApp()
    qt_app.processEvents()

    # get top-level widgets that aren't hidden
    widgets = [w for w in qt_app.topLevelWidgets() if not w.isHidden()]

    rendered_imgs = []
    for widg, imgpath in zip(widgets, imgpath_iter):
        pixmap = widg.grab()
        pixmap.save(imgpath)
        rendered_imgs.append(imgpath)
        widg.close()

    return scrapers.figure_rst(rendered_imgs, gallery_conf['src_dir'])

def do_nothing():
    return None

def reset_qapp(one, two):
    global qt_app
    qt_app.exec_ = do_nothing  # Kill the exec_ method to avoid blocking

def start_display(app, config):
    global display
    display = Display(backend="xvfb", size=(800, 600))
    display.start()

def stop_display(app, exception):
    # seems to be necessary to avoid "fatal IO error on X server..."
    reset_qapp(None, None)
    if display is not None:
        display.stop()

def setup(app):
    app.connect("config-inited", start_display)
    app.connect("build-finished", stop_display)
    return {"version": "0.1"}

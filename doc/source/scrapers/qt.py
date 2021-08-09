from sphinx_gallery import scrapers
import pyqtgraph as pg
from pyvirtualdisplay import Display

global app
app = pg.mkQApp()

disp = None

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

    global app
    app = pg.mkQApp()
    app.processEvents()

    # get top-level widgets that aren't hidden
    widgets = [w for w in app.topLevelWidgets() if not w.isHidden()]

    rendered_imgs = []
    for widg, imgpath in zip(widgets, imgpath_iter):
        pixmap = widg.grab()
        pixmap.save(imgpath)
        rendered_imgs.append(imgpath)
        widg.close()

    return scrapers.figure_rst(rendered_imgs, gallery_conf['src_dir'])

def reset_qapp(gallery_conf, fname):
    """Shutdown an existing QApplication and disable ``exec_``.

    Disabling ``QApplication.exec_`` means your example scripts can run the Qt event
    loop (so the scripts work when run normally) without blocking example execution by
    sphinx-gallery.

    Intended for use in ``image_scrappers`` in the sphinx-gallery configuration.
    """
    global app
    app.exec_ = lambda _: None

def setup(app):
    from .qtgallery import setup
    return setup(app)

def start_display(app, config):
    global disp
    disp = Display(backend="xvfb", size=(800, 600))
    disp.start()

def stop_display(app, exception):
    # seems to be necessary to avoid "fatal IO error on X server..."
    reset_qapp(None, None)
    if disp is not None:
        disp.stop()

def setup(app):
    app.connect("config-inited", start_display)
    app.connect("build-finished", stop_display)
    return {"version": "0.1"}

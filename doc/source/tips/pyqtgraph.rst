.. _pyqtgraph_anchor:

pyqtgraph
=========

The |pyqtgraph| package is great for interacting with data.  It has many nice features that are not found in
|matplotlib|.  However, it is not nearly as well documented as |matplotlib|.  Here are some common things we do with
|pyqtgraph|.

Keeping a window open
---------------------

If you write a script and your pyqtgraph window closes, try this:

.. code-block:: python

    pg.image(data)
    pg.QtGui.QApplication.instance().exec_()

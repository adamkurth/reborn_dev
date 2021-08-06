reborn Home
===========

Welcome to the home page of the reborn python package.

Overview of reborn package
--------------------------

The reborn Python package contains utilities for the simulation and analysis of x-ray diffraction under the Born 
approximation.  There are countless other utilities with related aspirations:
`thor <https://github.com/tjlane/thor>`_,
`OnDa <https://github.com/ondateam>`_,
`DIALS <https://dials.github.io/>`_,
`cctbx <https://cci.lbl.gov/cctbx_docs/index.html#id2>`_,
`psgeom <https://github.com/slaclab/psgeom>`_,
`xrayutilities <https://xrayutilities.sourceforge.io/index.html>`_,
`xraylib <https://github.com/tschoonj/xraylib/wiki>`_,
`Dragonfly <https://github.com/duaneloh/Dragonfly/wiki/EMC-implementation>`_,
`BornAgain <www.rebornproject.org>`_.

For clarity, reborn is not a "program".  In order to use it you must write Python code.  It is not by any measure
a *general* or *complete* tool at this time; it's content is presently dictated primarily by what is needed for work
done by the Kirian group.  It is nonetheless available for anyone to use under the GNU V3
License.

The reborn package is under constant development.  However, we do try to maintain useful and up-to-date documentation
along with an extensive suite of tests to make sure that it is reasonably stable.  The icon below indicates if tests are
presently passing in the master branch:

.. image:: https://gitlab.com/kirianlab/reborn/badges/master/pipeline.svg

You can find the reborn source code on gitlab here: https://gitlab.com/kirianlab/reborn

This documentation is available on the web here: https://kirianlab.gitlab.io/reborn

What's in reborn?
-----------------

In a nutshell, the basic elements reborn are:

- Classes for describing incident x-ray beams.
- Classes for describing detector geometries.
- Classes for describing objects that we shoot with x-rays.
- GPU-based simulation utilities.
- Tools for reading/writing a few file formats.
- Tools for displaying diffraction data.
- A few analysis algorithms.

For Newbies who want to use reborn..
------------------------------------

- If you are totally new to writing data analysis or simulation code, do learn the basics of
  `shell scripting <https://linuxconfig.org/bash-scripting-tutorial-for-beginners>`_,
  `git <https://www.vogella.com/tutorials/Git/article.html>`_,
  `python <https://becksteinlab.physics.asu.edu/learning/48/learning-python>`_.
- Learn how to use the |numpy| python package.
- When you learn python, make sure you learn object-oriented programming concepts.
- Skim through all of the reborn docs so you know what is here.  Complaints are not allowed until you've done that!
- If documentation is missing or confusing, please fix the problem or notify someone who can.
- The units in reborn are SI.  Angles are radians.  You rarely need to convert units within reborn.
- No special coordinate system is assumed in reborn.  You specify the direction of the x-ray beam.
- We make consistent assumptions about the shapes and memory layout of |numpy| arrays.
- The :ref:`examples` page is a good place to learn about how we use reborn.

If you plan to develop reborn...
--------------------------------

See :ref:`developers_anchor`.

Acknowledgements
----------------

The reborn package is maintained by `Rick Kirian <https://www.physics.asu.edu/content/richard-kirian>`_
with contributions from Derek Mendez, Joe Chen, Kevin Schmidt, Kosta Karpos, Roberto Alvarez, Rick Hewitt, and
Cameron Howard.  Code found in reborn has been inspired by numerous open-source software packages listed above.
Development is supported by National Science Foundation awards
`1231306 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1231306>`__,
`1943448 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1943448>`__,
and `1565180 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1565180>`__.

Contents
--------

.. toctree::
   :maxdepth: 1

   self
   installation
   theory
   beams
   geometry
   targets
   simulations
   examples
   tips
   developers
   api/modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

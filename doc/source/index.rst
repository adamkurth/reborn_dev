reborn Home
===========

.. image:: https://gitlab.com/kirianlab/reborn/badges/master/pipeline.svg

Overview of reborn package
--------------------------

The reborn Python package contains utilities for the simulation and analysis of x-ray diffraction under the Born 
approximation.  There are countless other utilities with identical or similar aspirations: 
`thor <https://github.com/tjlane/thor>`_,
`OnDa <https://github.com/ondateam>`_,
`DIALS <https://dials.github.io/>`_,
`cctbx <https://cci.lbl.gov/cctbx_docs/index.html#id2>`_,
`psgeom <https://github.com/slaclab/psgeom>`_,
`xrayutilities <https://xrayutilities.sourceforge.io/index.html>`_,
`xraylib <https://github.com/tschoonj/xraylib/wiki>`_,
`Dragonfly <https://github.com/duaneloh/Dragonfly/wiki/EMC-implementation>`_,
`BornAgain <www.rebornproject.org>`_.

For clarity, reborn is not a "program".  In order to use it you must write Python code.  It is also not intended
to be a *general* or *complete* tool at this time -- its content is dictated by what is needed for work done in the
Kirian Lab.  It is nonetheless available for anyone to use under the GNU V3 License.

You can find the reborn source code on gitlab here: https://gitlab.com/kirianlab/reborn

This documentation is available on the web here: https://kirianlab.gitlab.io/reborn

If you have never written code to do data analysis or simulations, you should firstly learn the basics of using the
`bash shell <https://linuxconfig.org/bash-scripting-tutorial-for-beginners>`_,
`git <https://www.vogella.com/tutorials/Git/article.html>`_,
and of course `python <https://becksteinlab.physics.asu.edu/learning/48/learning-python>`_.

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

Before you start using reborn...
--------------------------------

- Make sure you know how to use bash, git, and python.
- Make sure you at least skim through all of the reborn docs.  Complaints are not allowed until you've done that!
- If documentation is missing or confusing, please fix the problem or notify someone who can.
- The units in reborn are SI.  Angles are radians.  You rarely need to convert units within reborn.
- No special coordinate system is assumed.  You specify the direction of the x-ray beam.
- We make important assumptions about the shapes and memory layout of numpy arrays in reborn.  Read the docs.

If you plan to develop reborn...
--------------------------------

See :ref:`developers_anchor`.

Acknowledgements
----------------

The reborn package is maintained by Rick Kirian (rkirian at asu dot edu) with conbributions from Derek Mendez,
Joe Chen, Kevin Schmidt, Rick Hewitt, and Cameron Howard.  Code found in reborn has been inspired by numerous
open-source software packages listed above.

Contents
--------

.. toctree::
   :maxdepth: 1

   self
   installation
   beams
   geometry
   targets
   numpy
   crystals
   sampling_binning
   simulations
   fortran
   examples
   developers
   api/modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

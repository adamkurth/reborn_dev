(Re)reborn Python Package
============================

Overview
--------

The (Re)reborn Python package is for the simulation and analysis of x-ray diffraction under the Born approximation.
The "again" part of the name refers to the overlap with countless other utilities: e.g.
`thor <https://github.com/tjlane/thor>`_,
`OnDa <https://github.com/ondateam>`_,
`DIALS <https://dials.github.io/>`_,
`cctbx <https://cci.lbl.gov/cctbx_docs/index.html#id2>`_,
`psgeom <https://github.com/slaclab/psgeom>`_,
`xrayutilities <https://xrayutilities.sourceforge.io/index.html>`_,
`xraylib <https://github.com/tschoonj/xraylib/wiki>`_,
`Dragonfly <https://github.com/duaneloh/Dragonfly/wiki/EMC-implementation>`_,
and so on.  The "(Re)" part of the name is technically meant to distinguish from the similarly-named
`BornAgain <www.rebornproject.org>`_ software that also concerns diffraction and the Born approximation, but from
this point on we'll use our original name "reborn" (no capitalization) to refer to "(Re)reborn".

For clarity, reborn is not a "program".  In order to use it you must write Python code.  It is also not intended
to be a *general* or *complete* tool at this time -- its content is dictated by what is needed for work done in the
Kirian Lab.  It is nonetheless available for anyone to use under the GNU V3 License.

You can find the reborn source code on gitlab here: https://gitlab.com/rkirian/reborn

This documentation is available on the web here: https://rkirian.gitlab.io/reborn

What's in reborn?
--------------------

In a nutshell, the basic elements reborn are:

- Classes for describing incident x-ray beams.
- Classes for describing detector geometries.
- Classes for describing objects that we shoot with x-rays.
- GPU-based simulation utilities.
- Tools for reading/writing a few file formats.
- Tools for displaying diffraction data.
- A few analysis algorithms.

Before you start using reborn...
-----------------------------------

- Make sure you at least skim the docs.  Complaints are not allowed until you've done that!
- If documentation is missing or confusing, fix the problem or notify someone who can.
- The units in reborn are SI.  Angles are radians.  You rarely need to convert units within reborn; the only
  exceptions so far are a couple of low-level functions that convert PDB and CrystFEL geom files to Python dictionaries.
  Any reborn class or analysis function must use the standard units.
- No special coordinate system is assumed.  You specify the direction of the x-ray beam and so on.
- We make important assumptions about the shapes and memory layout of numpy arrays in reborn.  You'll learn about
  that when you read the docs.

If you plan to develop reborn...
-----------------------------------

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

   installation
   beams
   geometry
   targets
   numpy
   crystals
   sampling_binning
   simulations
   examples
   fortran
   developers
   api/modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

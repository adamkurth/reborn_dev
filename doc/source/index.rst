(Re)bornagain Python Package
============================

Overview
--------

The (Re)bornagain Python package is for the simulation and analysis of x-ray diffraction under the Born approximation.
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
`BornAgain <www.bornagainproject.org>`_ software that also concerns diffraction and the Born approximation, but from
this point on we'll use our original name "bornagain" (no capitalization) to refer to "(Re)bornagain".

For clarity, bornagain is not a "program".  In order to use it you must write Python code.  It is also not intended
to be a *general* or *complete* tool at this time -- its content is dictated by what is needed for work done in the
Kirian Lab.  It is nonetheless available for anyone to use under the GNU V3 License.


What's in bornagain?
--------------------

In a nutshell, the basic elements bornagain are:

- Classes for describing incident x-ray beams.
- Classes for describing detector geometries.
- Classes for describing objects that we shoot with x-rays.
- GPU-based simulation utilities.

In the future, we will add:

- Tools for reading/writing a few file formats.
- Tools for displaying diffraction data.
- A few analysis algorithms.

Before you start using bornagain...
-----------------------------------

- Make sure you at least skim the docs.  Complaints are not allowed until you've done that!
- If documentation is missing or confusing, fix the problem or notify someone who can.
- The units in bornagain are SI.  Angles are radians.  You rarely need to convert units within bornagain; the only
  exceptions so far are a couple of low-level functions that convert PDB and CrystFEL geom files to Python dictionaries.
  Any bornagain class or analysis function must use the standard units.
- No special coordinate system is assumed.  You specify the direction of the x-ray beam and so on.
- We make important assumptions about the shapes and memory layout of numpy arrays in bornagain.  You'll learn about
  that when you read the docs.

If you plan to develop bornagain...
-----------------------------------

See the page for developers.  See :ref:`developers_anchor`.

Acknowledgements
----------------

Code found in bornagain has been inspired by numerous open-source software packages listed above.  Contributions to
bornagain have been made by

- Derek Mendez
- Joe Chen
- Rick Hewitt
- Cameron Howard

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
   fortran
   developers
   api/modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

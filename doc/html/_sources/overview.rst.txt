Overview
========

The bornagain Python package is meant to be used for the simulation and analysis of
x-ray diffraction under the Born approximation.  It is not the first attempt to create
such a package, hence the name.

It turns out that there is another Python package called "`BornAgain <www.bornagainproject.org>`_"... and it is for
simulating diffraction under the Born approximation (this package will probably be re-named some day...).  There are
of course lots of other Python packages that provide utilities that overlap in various ways with bornagain: e.g.
`thor <https://github.com/tjlane/thor>`_,
`OnDa <https://github.com/ondateam>`_,
`DIALS <https://dials.github.io/>`_,
`cctbx <https://cci.lbl.gov/cctbx_docs/index.html#id2>`_,
`psgeom <https://github.com/slaclab/psgeom>`_,
`xrayutilities <https://xrayutilities.sourceforge.io/index.html>`_,
`xraylib <https://github.com/tschoonj/xraylib/wiki>`_,
`Dragonfly <https://github.com/duaneloh/Dragonfly/wiki/EMC-implementation>`_
and so on.


What's in bornagain?
--------------------

For clarity, bornagain is not a "program" - in order to use it you must write Python code.  In a nutshell, the basic
elements of the bornagain package are:

- Classes for describing incident x-ray beams.
- Classes for describing detector geometries.
- Classes for describing objects that we shoot with x-rays.
- GPU-based simulation utilities.

In the future, we will add a few more utilities to bornagain:

- Tools for reading/writing a few file formats.
- Tools for displaying diffraction data.
- Analysis algorithms.

Before you start using bornagain
--------------------------------

- If documentation is missing or confusing, please fix it or tell someone who can.
- *All* units in bornagain are SI.  Angles are radians.  No exceptions (so far).
- No special coordinate system is assumed.  That is, *You* get to choose the direction of the x-ray beam and so on.

If you plan to develop bornagain
--------------------------------

See the page for developers :ref:`developers_anchor`.

Acknowledgements
----------------

Code found in bornagain has been inspired by numerous open-source software packages listed above.  Numerous
contributions to bornagain have been made by Derek Mendez, Rick Hewitt, Cameron Howard, and Joe Chen.

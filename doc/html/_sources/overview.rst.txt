Overview
========

The bornagain Python package is meant to be used for the simulation and analysis of
x-ray diffraction under the Born approximation.  It is not the first attempt to create
such a package, hence the name.

It turns out that there is another Python package called "`BornAgain <www.bornagainproject.org>`_"... and it is for
simulating diffraction under the Born approximation.  Since they were first (and their software looks really impressive)
this package will probably be re-named...

Since bornagain is under continuous development, there might still be some large changes to the API in the near future.

There are lots of other Python packages that provide utilities that overlap with bornagain:

- DIALS
- cctbx
- psgeom
- poppy
- xrayutilities
- xraylib
- and so on


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


I don't know anything about programming...
------------------------------------------

If you are new to programming, you should first:

* Learn to program in the `Python <https://www.python.org/>`_ language,
* Learn the basics of object-oriented programming,
* Master the use of the `numpy <http://www.numpy.org/#>`_ package.

There are some example scripts in the bornagain/examples directory that might help you get started. You should
tinker with those scripts using `iPython <https://ipython.org/>`_ and its
`tab-completion feature <https://ipython.org/ipython-doc/3/interactive/tutorial.html#tab-completion>`_.


Before you start using bornagain
--------------------------------

- If documentation is missing or confusing, please fix it or tell someone who can.
- *All* units in bornagain are SI.  Angles are radians.  No exceptions.
- There is no special x-ray beam direction.  *You* get to choose the direction of the x-ray beam.
- We must be consistent in the way that we specify vectors and rotation matrices using numpy arrays.  There are
  utilities to ensure consistency.


If you plan to develop bornagain
--------------------------------

See the page for developers :ref:`developers_anchor`.  We aim to keep things reasonably consistent.

Acknowledgements
----------------

There are numerous contributions to bornagain made by Derek Mendez, Rick Hewitt, and Cameron Howard.

Code found in bornagain has been inspired by numerous open-source software packages such as Thor, psana, OnDA, CrystFEL,
Cheetah, cctbx.  In some cases, code has been copied or paraphrased from other open-source Python modules such as spglib
and cfelpyutils.
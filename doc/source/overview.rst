Overview
========

The bornagain Python package is meant to be used for the simulation and analysis of
x-ray diffraction under the Born approximation.  It is not the first attempt to create
such a package, hence the double-meaning of the name.

For clarity, bornagain is not a "program".  To use a *Python package* such as bornagain, you must
write your own custom python scripts or programs.

**Beware**: bornagain is under continuous development; there might still be some large
changes to the API in the near future.

What's in bornagain?
--------------------

In a nutshell, the basic elements of the bornagain package are:

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

If you are new to programming, the following might help:

* You *must* learn the `Python <https://www.python.org/>`_ language.
* You *should* learn the basics of object-oriented programming.
* You should master the use of the `numpy <http://www.numpy.org/#>`_ package.
* There are some example scripts to help you get started.  Look in the bornagain/examples directory.
* `iPython <https://ipython.org/>`_ (and its
  `tab-completion feature <https://ipython.org/ipython-doc/3/interactive/tutorial.html#tab-completion>`_)
  is a great way to explore bornagain.

Before you start using bornagain
--------------------------------

- If documentation is missing or confusing, please fix it or tell someone who can.
- *All* units in bornagain are SI.  Angles are radians.
- We never hard-code the direction the x-ray beam.  *You* choose the direction of the x-ray beam.
- We must be consistent in the way that we specify vectors using numpy arrays.  Same goes for rotation matrices.  We therefore have utilities like vec_check() and rotate_vecs() to make sure we shape arrays consistently and operate on vectors in a consistent way.  Use these utilities.


If you plan to develop bornagain
--------------------------------

See the page for developers.

Acknowledgements
----------------

There are numerous contributions to bornagain made by Derek Mendez, Rick Hewitt, and Cameron Howard.

Code found in bornagain has been inspired by numerous open-source software packages such as Thor, psana, OnDA, CrystFEL,
Cheetah, cctbx.  In some cases, code has been copied or paraphrased from other open-source Python modules such as spglib
and cfelpyutils.
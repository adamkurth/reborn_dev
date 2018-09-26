Overview
========

What's in bornagain?
--------------------

In a nutshell, the basic elements of the bornagain package consist of:

- Source classes for describing incident x-ray beams.
- Detector classes for specifying detector geometry.
- Target classes that describe objects that we shoot with x-rays.
- GPU-based simulation utilities.

In the future, we will add a few more utilities to bornagain:

- Tools for dealing with reading/writing of a few file formats.
- Tools displaying diffraction data.
- Analysis algorithms.


I don't know anything about programming...
------------------------------------------

If you are new to programming, read the following:

- In order to make good use of bornagain, you must learn the `Python <https://www.python.org/>`_ language.  You probably also need to learn the basics of object-oriented programming.
- There are some example scripts to help you get started.  Look in the bornagain/examples directory.
- `iPython <https://ipython.org/>`_ (and its `tab-completion feature <https://ipython.org/ipython-doc/3/interactive/tutorial.html#tab-completion>`_ is a good way to explore bornagain.
- The `numpy <http://www.numpy.org/#>`_ package is central to bornagain.  You must learn to use numpy along with the basic principles of `array programming <https://en.wikipedia.org/wiki/Array_programming>`_.


Before you get started...
-------------------------

- The Python language is the interface to bornagin.  It's very readable if you `try <https://www.python.org/dev/peps/pep-0020/>`_ to make it so.
- All units in bornagain are SI, for simplicity.  Angles are radians.
- We must be consistent in the way that we specify vectors using numpy arrays.  Same goes for rotation matrices.  We therefore have utilities like vec_check() and rotate_vecs() to make sure we shape arrays consistently and operate on vectors in a consistent way.  Use these utilities.
- We never hard-code the direction the x-ray beam.  You specify the direction that *you* want.
- We adhere to `PEP8 <https://www.python.org/dev/peps/pep-0008/?>`_ Python style guide.
- If documentation is missing or confusing, please fix it or tell someone who can.


If you plan to develop bornagain
--------------------------------

See the page for developers.

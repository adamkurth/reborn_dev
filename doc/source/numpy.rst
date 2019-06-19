Numpy, vectors, rotations, etc.
===============================

We use numpy a lot, and there are a few important conventions within bornagain to be aware of.


Indexing and internal memory layout of ndarray objects
------------------------------------------------------

According to the `docs <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray>`_,
Numpy ndarray objects can accommodate any strided indexing scheme.  The bornagain package uses some
Fortran and OpenCL routines that make assumptions about the internal memory layout of ndarrays.  We assume that
all ndarrays are in the "c-contiguous order", which means that incrmements in the right-most index correspond to the
smallest steps in the internal memory buffer.  The following example illustrates this:

.. code-block:: python

    import numpy as np
    a = np.array([[1,2,3],[4,5,6]])
    print(a.shape)
    (2, 3)
    print(a.flags.c_contiguous)
    True
    print(a.ravel())
    [1 2 3 4 5 6]

Arrays of vectors
-----------------

If you have *N* vectors of dimension *D*, bornagain assumes they are stored with a shape of (*N*, *D*).  This choice was
made because the right-most index of a numpy array has the smallest stride by default, and because it usually makes
most sense to have vector components stored close to each other in memory.  This convention is assumed in every function
in bornagain.  This note is to avoid ambiguity in the case of a (*D*, *D*) array.

How to rotate vectors
---------------------

If you need to rotate a vector or an array of vectors with the matrix *R*, you can do the following

.. code-block:: python

    vec_rotated = np.dot(R, vec.T).T

The above is equivalent to:

.. code-block:: python

    vec_rotated = np.dot(vec, R.T)

You can also use the :func:`rotate <bornagain.utils.rotate>` function for consistency.  Here is what you should expect:

.. code-block:: python

    R = np.array([[0, 1., 0], [-1, 0, 0], [0, 0, 1.]])
    vec = np.array([1, 2, 3])
    vec_rotated = utils.rotate(R, vec)
    print(vec_rotated)

      [ 2. -1.  3.]

Note that the :func:`rotate <bornagain.utils.rotate>` function carried out via numpy is consistent with rotation
operations performed on GPU devices within the :mod:`simulate.clcore <bornagain.simulate.clcore>` module.

Representation of 3D density/intensity maps in numpy arrays
-----------------------------------------------------------

This needs to be specified some day...
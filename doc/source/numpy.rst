Numpy, vectors, rotations, etc.
===============================

We use numpy a lot, and there are a few important conventions within bornagain to be aware of.


About indexing and memory
-------------------------

By default, increments in the right-most index of a numpy array correspond to the smallest increments
in the memory buffer.  In other words, the right-most index has the shortest stride.  Some people refer to this as
the "row-major" format, as opposed to the "column-major" format whereby the left-most index corresponds to
the shortest stride.  (If these concepts are new to you, make sure you do the relevant background reading -- perhaps
`this <https://www.jessicayung.com/numpy-arrays-memory-and-strides/>`_ helps).  Importantly, when certain operations are
performed on a numpy array, a row-major array might become a column-major array.  Other operations in numpy might cause
the memory buffer to be wiped out and re-written.  It is also common that the modification of an array might affect
another another one, if the two have a shared memory buffer.  Basically, you need to be careful with how you use
numpy arrays if this matters, and in fact it does matter when moving numpy arrays into memory on a GPU or when
interfacing with C or Fortran code.


How to store many vectors in a numpy array
------------------------------------------

If you have *N* vectors of dimension *D*, you should store them in a numpy array of shape *N* x *D*.  This choice was
made because the right-most index of a numpy array has the smallest stride by default, and because it usually makes
most sense to have vector components stored close to each other in memory.  This convention is assumed in every function
in bornagain.

How to rotate vectors
---------------------

The short answer is to store your set of *N* vectors of dimension *D* as an *N* x *D* numpy array, and then apply your
*D* x *D* rotation matrix via the :func:`rotate <bornagain.utils.rotate>` function.  Here is what you should expect:

.. code-block:: python

    R = np.array([[0, 1., 0], [-1, 0, 0], [0, 0, 1.]])
    vec = np.array([1, 2, 3])
    vec_rotated = utils.rotate(R, vec)
    print(vec_rotated)

      [ 2. -1.  3.]

The :func:`rotate <bornagain.utils.rotate>` function simply does the following:

.. code-block:: python

    vec_rotated = np.dot(R, vec.T).T

The above is equivalent to:

.. code-block:: python

    vec_rotated = np.dot(vec, R.T)

Interestingly, the former method is faster than the latter.

Note that the :func:`rotate <bornagain.utils.rotate>` function carried out via numpy is consistent with rotation
operations performed on GPU devices within the :mod:`simulate.clcore <bornagain.simulate.clcore>` module.

How to store many 3D density/intensity maps in numpy arrays
-----------------------------------------------------------

This needs to be specified some day...
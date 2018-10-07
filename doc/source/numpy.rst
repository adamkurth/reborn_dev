Numpy, vectors, rotations, etc.
===============================

We use numpy a lot, and there are a few important conventions within bornagain to be aware of.

How to store many vectors in a numpy array
------------------------------------------

If you have *N* vectors of dimension *D*, you should store them in a numpy array of shape *N* x *D*.  If you do something
different, it is possible that rotations (for example) will be performed incorrectly.  This choice of convention was
made because the right-most index of a numpy array has the smallest stride (by default -- this can be changed, but that
would create a new set of bigger problems...), and because it seems to make most sense to have vector components stored
close to each other in memory.  This is important, for example, when we pass numpy arrays to opencl for fast GPU
simulations.

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

Interestingly, the former method is faster than the latter, despite the fact that there are two transpose operations.
Also noteworthy is the fact that the `np.dot` function is fastest if we could avoid all transpose operations.  Maybe
we can do that, but I don't see how.

Note that the :func:`rotate <bornagain.utils.rotate>` function carried out via numpy is consistent with rotation
operations performed on GPU devices within the :mod:`simulate.clcore <bornagain.simulate.clcore>` module.

How to store many 3D density/intensity maps in numpy arrays
-----------------------------------------------------------

For 3D density/intensity maps, we choose to index such that the "z" coordinate corresponds to the right-most numpy
index.
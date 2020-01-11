.. _numpy_anchor:

Working with Numpy
==================

Indexing and internal memory layout of ndarray objects
------------------------------------------------------

According to the
`docs <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray>`_,
Numpy ndarray objects can accommodate any strided indexing scheme.  The bornagain package uses some
Fortran and OpenCL routines that make assumptions about the internal memory layout of ndarrays, which means that we
need to be clear about how we store our arrays in memory.  We assume that
all ndarrays are in the *default* "c-contiguous order", which defines a syntax in which incremements in the right-most
index of an array correspond to the smallest steps in the internal memory buffer.  The data buffer is contiguous --
there are no "gaps".  The following example illustrates this:

.. testcode::

    import numpy as np
    a = np.array([[1,2,3],[4,5,6]])
    print(a.shape)
    print(a.flags.c_contiguous)
    print(a.ravel())
    print(a[0, 1])

.. testoutput::

    (2, 3)
    True
    [1 2 3 4 5 6]
    2

Matrices
--------

Numpy has a matrix class but it is not recommended to use it (according to the numpy docs).  So we use regular numpy
arrays to store matrices.  In order to take a matrix product, we use the np.dot function as illustrated in the following
example:

.. testcode::

    import numpy as np
    A = np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
    B = np.array([[1, 0, 0],[0, 2, 0],[0, 0, 3]])
    AB = np.dot(A, B)
    print(A)
    print(B)
    print(AB)

.. testoutput::

    [[ 0  1  0]
     [-1  0  0]
     [ 0  0  1]]
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    [[ 0  2  0]
     [-1  0  0]
     [ 0  0  3]]

Arrays of vectors
-----------------

.. _arrays_of_vectors:

If you have *N* vectors of dimension 3, bornagain assumes they are stored with a shape of (*N*, 3).  This choice was
made because the right-most index of a numpy array has the smallest stride by default, and because it usually makes
most sense to have vector components stored close to each other in memory.  This convention is assumed in every function
in bornagain.  This note is to avoid ambiguity in the case of a (3, 3) array.

Rotations
---------

We must be certain that we adhere to a convention with regard to vector rotations.  If you need to rotate a vector or an
array of vectors with shape (*N*, 3) with the matrix *R*, you can do either of the following:

.. testcode::

    import numpy as np
    R = np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
    vec = np.array([[1, 2, 3], [4, 5, 6]])
    v1 = np.dot(R, vec.T).T
    v2 = np.dot(vec, R.T)
    assert np.all(v1 == v2)

For clarity, here is what you should expect:

.. testcode::

    R = np.array([[0, 1., 0], [-1, 0, 0], [0, 0, 1.]])
    vec = np.array([1, 2, 3])
    vec_rotated = np.dot(vec, R.T)
    print(R)
    print(vec)
    print(vec_rotated)

.. testoutput::

    [[ 0.  1.  0.]
     [-1.  0.  0.]
     [ 0.  0.  1.]]
    [1 2 3]
    [ 2. -1.  3.]


Note that the above is consistent with rotation operations performed on GPU devices within the
:mod:`simulate.clcore <bornagain.simulate.clcore>` module.

Density maps
------------

As with vectors, it is also important that we have an understanding of how to represent density maps as numpy arrays.
In particular, we need to be clear on how we assign positional coordinate vectors to elements in the density arrays.
This is discussed in the :ref:`density map <nd_array_handling>` page.
.. _numpy_anchor:

Numpy
=====

Numpy is the most important Python package that reborn uses when working with arrays.  Below are a few declarations
that are meant to resolve possible ambiguities when using numpy arrays within the reborn framework.  Users who are
new to numpy should study it -- there are countless tutorials on the web.

Indexing and internal memory layout of ndarray objects
------------------------------------------------------

*People who are familiar with numpy ndarrays and their memory layout can skip this section -- the synopsis is simple:
some parts of the reborn package (particularly those that have underlying Fortran or OpenCL code) assume that
ndarrays are are in the default c-contiguous ordering.  There are some cases in which errors will result if you pass
in an array that is not c-contiguous, and in other cases arrays are re-written before passing to Fortran or OpenCL
functions, which will cause in speed reductions.*

Some users of numpy can carry out all of their Python/numpy calculations without knowledge of the internal memory
structure of numpy ndarray objects.
However, most will eventually learn that it is essential to have at least some basic knowledge of how numpy actually
handles the underlying array data in the computer memory.
A simple example illustrates the need for this:

.. testcode::

    import numpy as np
    a = np.ones(3)
    b = a
    print(b)
    a[0] = 0
    print(b)

.. testoutput::

    [1. 1. 1.]
    [0. 1. 1.]

From the output it is evident that the underlying memory buffer of the *a* array is the same as the *b* array; you might
say that *by changing a we also changed b*.
Presumably, the reason for this behavior is that ndarrays are designed to avoid unnecessary time-consuming memory
(re)writes.
The numpy ndarray class is also meant to provide a clean interface that removes the need to direcly manipulate memory so
that you can write programs faster, but this apparent simplicity comes at the cost of obscurity.
Students who begin using numpy without consideration of computer memory are often frustrated by this obscurity.

Now we make a small change to the previous example:

.. testcode::

    import numpy as np
    a = np.ones(3)
    b = a*1
    print(b)
    a[0] = 0
    print(b)

.. testoutput::

    [1. 1. 1.]
    [1. 1. 1.]

In this example the output shows that the *a* and *b* arrays no longer share a common memory buffer.
The reason is that the multiplication operation caused the ndarray object to create a new memory buffer for *b*.
There are many other examples that expose some of the behaviors of ndarrays that are related to memory, but we will not
venture further here since there are many tutorials on this topic on the web and also the
`numpy ndarray docs <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray>`_
have helpful information.
As a final note on the above examples, you should be aware that the *copy* function can be used in order to be certain
that two ndarrays have distinct memory, as in this example:

.. testcode::

    import numpy as np
    a = np.ones(3)
    b = a.copy()
    print(b)
    a[0] = 0
    print(b)

.. testoutput::

    [1. 1. 1.]
    [1. 1. 1.]

Additional issues can arise as a result of the fact that numpy ndarray objects can accommodate
`any strided indexing scheme <https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray>`_
.
By default, ndarrays are in the "c-contiguous order", which means that incremements in the right-most index of an
ndarray correspond to the smallest stride in the internal memory buffer.
However, there are many operations that result in ndarrays that are not in "c-contiguous" order.
The following example illustrates this:

.. testcode::

    import numpy as np
    a = np.arange(9).reshape([3, 3])
    print(a)
    print(a.flags.c_contiguous)
    b = a.T  # Transpose the a array
    print(b)
    print(b.flags.c_contiguous)
    print(b.flags.f_contiguous)
    a[0, 0] = 1
    print(b)

.. testoutput::

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    True
    [[0 3 6]
     [1 4 7]
     [2 5 8]]
    False
    True
    [[1 3 6]
     [1 4 7]
     [2 5 8]]

As we can see, the transpose operation reverses the ordering of the indices but does not modify the memory buffer.
The result is an ndarray with a memory buffer in "f-contiguous order" (the *first* index has the shortest stride).
In most cases, the ordering does not matter since virtually all of numpy is designed to be indifferent to
the layout of internal memory buffers.
You might notice that the speed of your program depends on the ordering of the internal memory, but you will probably
get the result you expect regardless of the ordering.

The central point in introducing the above is the following: some portions of the code in reborn are written in the
Fortran and OpenCL languages, and as a result *the ordering of the memory buffers matters for some functions in
reborn*.
In order to make this issue as painless as possible, it is assumed that all ndarrays are in the default
"c-contiguous" order, and the striding corresponds to contiguous data (there are no "gaps" between array elements).
There are more details on this matter found elsewhere (see e.g. :ref:`Working with Fortran <working_with_fortran>`).


Matrices
--------

Numpy has a matrix class but it is not recommended to use it (according to the numpy docs).
We use regular numpy arrays to store matrices.
Fortunately, numpy displays arrays as you would likely write them down mathematically, as shown in the following
example:

.. testcode::

    a = np.arange(9).reshape((3,3))
    print(a)
    print(a[0,1])

.. testoutput::

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    1

As you can see, the first index corresponds to "row index" and the second index corresponds to the "column index".
If you are performing e.g. rotations on vectors and you are uncertain of the ordering of array elements, you can print
an example array and make sure it looks the way you would write it down on paper.

For ndarrays, the ordinary product operation (a*b) does an element-by-element product.
In order to take a matrix product of two arrays, we use the np.dot function as in the following example:

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

If you have *N* vectors of dimension 3, reborn assumes they are stored with a shape of (*N*, 3).  This choice was
made because the right-most index of a numpy array has the smallest stride by default, and because it usually makes
most sense to have vector components stored close to each other in memory.
This convention is assumed in every function in reborn that deals with arrays of vectors.
Normally you would get a runtime error if you pass in an array of the wrong shape, due to mis-match dimensions, but
there will be no error in the case of a (3, 3) array.

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
:mod:`simulate.clcore <reborn.simulate.clcore>` module.

Density maps
------------

As with vectors, it is also important that we have an understanding of how to represent density maps as numpy arrays.
In particular, we need to be clear on how we assign positional coordinate vectors to elements in the density arrays.
This is discussed in the :ref:`density map <nd_array_handling>` page.
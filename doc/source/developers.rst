.. _developers_anchor:

Developers
==========

Before you modify any code:
---------------------------

* The "`Zen of Python <https://www.python.org/dev/peps/pep-0020/>`_" captures the essence of Python programming
  norms.  We follow these norms in reborn.
* Follow the `PEP8 guidelines <https://www.python.org/dev/peps/pep-0008/?>`_.
* One exception to PEP8: we allow lines to be 120 characters in length.
* Please use four spaces, not tabs.
* Write unit tests for any functionality you add.  We use |pytest| for this purpose.
* Document your code!  It is important to follow the
  `Google format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_ so that the html documentation is
  formatted correctly.
* Learn how to use `git <https://git-scm.com/book/en/v2>`_.
* Develop code in the git "develop" branch.  We merge the develop branch into (protected) master only after tests are
  known to pass.
* All units are SI (angles in radians) unless there is a *very* good reason to do something different.  Consistency
  helps avoid bugs.
* The scope of this project is diffraction under the Born approximation.  Don't stray far from this.

Checking for PEP8 compliance
----------------------------

You can additionally use the pep8 program to check for inconsistencies (install pep8 with pip or conda if need be).
In the base directory of the git repo, do this

.. code-block:: bash

    pep8 reborn
    
For simple errors like whitespace, you can use autopep8:

.. code-block:: bash

    autopep8 -i -a filename.py
    
For other problems you'll need to fix things by hand.  We aim to have no errors coming from the `pep8` program.

We also use `pylint <https://www.pylint.org/>`_ for code formatting.  You should occasionally check how well your code
conforms to pylint standards:

.. code-block:: bash

    pylint --max-line-length=120 filename.py

We do not strive to remove *all* complaints made by pylint.  There are some unreasonable complaints such as "too
many function arguments".  You may therefore wish to use the helper script ``developer/mylint.sh``, which turns off some
of the commonly annoying complaints.

Testing
-------

We use |pytest| to test the reborn codebase.  It is very simple to make a new test.

1) Create a file that has a name that begins with ``test_`` in the ``reborn/test`` directory
2) Within this file, write functions with names that begin with ``test_``
3) Within those functions, include `assert statements <https://wiki.python.org/moin/UsingAssertionsEffectively>`_.
4) Run |pytest| in the test directory, and all tests will run (or run it on a specific file).


Generation of documentation
---------------------------

Docstrings from within the python code automatically find their way into this documentation via
`Sphinx <http://www.sphinx-doc.org/en/master/>`_.  Please keep the formatting consistent by adhering to the
`Google format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_ for doc strings.  Here is an example of a
decently written doc string:

.. code-block:: python

    r"""
    Calculate diffraction amplitudes according to the sum:

    .. math::
        a_i = \sum_n f_n \exp(-i \vec{q}_i \cdot (\mathbf{R} \vec{r} + \vec{U}))

    Arguments:
        q (numpy/cl float array [N,3]): Scattering vectors.
        r (numpy/cl float array [M,3]): Atomic coordinates.
        f (numpy/cl complex array [M]): Complex scattering factors.
        R (numpy array [3,3]): Rotation matrix.
        U (numpy array): Translation vector.
        a (cl complex array [N]): Complex scattering amplitudes (if you wish to manage your own opencl array).
        add (bool): True means add to the input amplitudes a rather than overwrite the amplitudes.
        twopi (bool): True means to multiply :math:`\vec{q}` by :math:`2\pi` just before calculating :math:`A(q)`.
        n_r_chunks (int): Run in n batches of position vectors to avoid memory issues.

    Returns:
        (numpy/cl complex array [N]): Diffraction amplitudes.  Will be a cl array if there are input cl arrays.
    """

If you modify code and wish to update this documentation, the easiest way to do so is to run the script
``update-documentation.sh`` from within the ``doc`` directory.

Speeding up code with numba and f2py
------------------------------------

Numba is one way to speed up Python code in cases where there is not an existing numpy function.  It is used within
reborn in a few places and appears to be reasonably stable, though still lacking some very basic functionality.

.. _working_with_fortran:

Integration of Fortran and Numpy
--------------------------------

The f2py utility included with numpy makes it quite easy to integrate simple Fortran code with Numpy.  Typically,
we wish to pass memory buffers from numpy ndarrays into a Fortran subroutine, and we modify those buffers with Fortran.
There are some very annoying issues that can arise because the ways in which the Numpy package manipulates
the inernal memory buffers of ndarrays, which might surprise you.  These under-the-hood manipulations might be
harmless until the day you really care about operating directly on memory buffers. Examples of such complications can
be found in the example :ref:`plot_f2py`.

Another matter is the way that numpy arrays are passed to fortran routines when you use f2py.  The
`documentation <https://www.numpy.org/devdocs/f2py/python-usage.html>`_ states the following:

    "*In general, if a NumPy array is proper-contiguous and has a proper type then it is directly passed to wrapped
    Fortran/C function. Otherwise, an element-wise copy of an input array is made and the copy, being proper-contiguous
    and with proper type, is used as an array argument.*"

Given the above we've come up with the following recipe to avoid possible issues:

(1) Always work with the default C-contiguous ndarray memory layout in Python code.

(2) Use assert statements in function wrappers: e.g. assert a.flags.c_contiguous == True.

(3) Transpose ndarrays before passing them to Fortran routines.  This will *not* copy memory.

(4) In your Fortran code, simply reverse the ordering of your indices as compared to your Numpy code.

Although it may be inconvenient to reverse your indexing when going between the Fortran and Python code, bear in mind
that this can only be avoided by (a) making copies of array memory, or (b) enforcing a consistent non-default internal
memory layout for all Numpy arrays that touch a Fortran routine.  Both options (a) and (b) are highly undesirable.  We
choose option (c), reverse the index order, because it holds the big advantage that we get to think about memory in the
most natural way for both Numpy *and* Fortran coding, rather than insisting that Fortran and Numpy syntax *look* the
same at the expense of speed and potential memory issues.

.. _developers_anchor:

Developers
==========

Before you modify any code:
---------------------------

* The "`Zen of Python <https://www.python.org/dev/peps/pep-0020/>`_" captures the essence of Python programming
  norms.  Please follow them for the sake of consistency.
* Follow the `PEP8 guidelines <https://www.python.org/dev/peps/pep-0008/?>`_.
* One exception to PEP8: we allow lines to be 120 characters in length.
* Use four spaces, not tabs.  No exceptions.
* Write unit tests for any non-trivial functionality you add.  We use |pytest| for this purpose.
* Document your code using the
  `Google format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_.
* Learn how to use `git <https://git-scm.com/book/en/v2>`_ if you haven't already.
* Develop code in the git "develop" branch.  Rick merges the develop branch into (protected) master branch only after
  all tests are passing.
* All units are SI (angles in radians) for the sake of consistency.
* The scope of this project is diffraction under the Born approximation.  Don't stray very far from this.

Checking for PEP8 compliance
----------------------------

We use |pycodestyle| and `pylint <https://www.pylint.org/>`_ to check for basic compliance with PEP8.  The script
``developer/pycodestyle.sh`` should be used to check for compliance.


Testing
-------

We use |pytest| to test the reborn codebase.  It is very simple to make a new test.

1) Create a file that has a name that begins with ``test_`` in the ``reborn/test`` directory
2) Within this file, write functions with names that begin with ``test_``
3) Within those functions, include `assert statements <https://wiki.python.org/moin/UsingAssertionsEffectively>`_.
4) Run |pytest| in the test directory.


Generation of documentation
---------------------------

Docstrings from within the python code automatically find their way into this documentation via
`Sphinx <http://www.sphinx-doc.org/en/master/>`_.  Please keep the formatting consistent by adhering to the
`Google format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_ for doc strings.  Here is an example of a
decently written doc string:

.. code-block:: python

    r"""
    Some basic description at the top.  You might link to other documentation inside of reborn, such as
    :class:`reborn.detector.PADGeometry` .  Some classes have shortcuts defined in ``doc/conf.py``, such as
    |PADGeometry|.  You can also link to exernal code docs, for example :func:`np.median`.

    Here is a random, unrelated equation for the purpose of demonstration:

    .. math::
        a_i = \sum_n f_n \exp(-i \vec{q}_i \cdot (\mathbf{R} \vec{r} + \vec{U}))

    .. note::
        This is a note, to emphasize something important.  For example, note that the return type is a tuple in this
        example, which requires special handling if you wish to specify the contents as if there are multiple returns.

    Arguments:
        data (|ndarray| or list of |ndarray|): Data to get profiles from.
        beam (|Beam|): Beam info.
        pad_geometry (|PADGeometryList|): PAD geometry info.
        mask (|ndarray| or list of |ndarray|): Mask (one is good, zero is bad).
        n_bins (int): Number of radial bins.
        q_range (tuple of floats): Centers of the min and max q bins.
        statistic (function): The function you want to apply to each bin (default: :func:`np.mean`).

    Returns:
        (tuple):
            - **statistic** (|ndarray|) -- Radial statistic.
            - **bins** (|ndarray|) -- The values of q at the bin centers.
    """

If you modify code and wish to update this documentation, the easiest way to do so is to run the script
``update_docs.sh`` from within the ``doc`` directory.

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

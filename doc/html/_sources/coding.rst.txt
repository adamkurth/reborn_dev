.. _developers_anchor:

Notes for Developers
====================

First and foremost, the bornagain package does not presently follow the guidelines below, but we are presently trying to
fix the many inconsistencies.  Please contribute to this effort!!!

Before you modify any code:
---------------------------

* Read the "`Zen of Python <https://www.python.org/dev/peps/pep-0020/>`_".
* Follow the `PEP8 guidelines <https://www.python.org/dev/peps/pep-0008/?>`_.
* One exception to PEP8: we allow lines to be 120 characters in length.
* Strive to write `pytest <http://doc.pytest.org/>`_ unit tests for any functionality you add.
* *Always* write docstrings.  Follow the `Google format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_.
* Learn how to use `git <https://git-scm.com/book/en/v2>`_.
* Familiarize yourself with our usage of git; we emulate `these ideas <https://nvie.com/posts/a-successful-git-branching-model/>`_.
* Use four spaces, not tabs.
* All units are SI (angles in radians) unless there is a *very* good reason to do something different.
* Just because you *can* do something clever, doesn't mean you should.  Don't confuse people.
* The scope of this project is diffraction under the Born approximation.  Don't stray far from this.

Helpful tools
-------------

* Look in the developer directory for helpful scripts such as "update-documentation.sh", which does what you guessed.
* Programs such as `pep8 <https://pypi.python.org/pypi/pep8/>`_ can be used to check for PEP8 compliance.
* Consider using an integrated development environment such as pycharm.

Python 2/3 compatibility
------------------------

To ensure compatibility with both Python 2 and 3, include the following at the beginning of each python file:

.. code-block:: python

    from __future__ import (absolute_import, division, print_function, unicode_literals)


Checking for PEP8 compliance
----------------------------

We are now using pylint for code formatting.  Install pylint with conda or pip if need be.  You should check that your
code conforms to standards as follows:

.. code-block:: bash

    pylint --max-line-length=120 filename.py

You can also use the pep8 program to check for inconsistencies (install pep8 with pip if need be).  In the
base directory of the git repo, do this

.. code-block:: bash

    pep8 bornagain
    
For simple errors like whitespace, you can use autopep8:

.. code-block:: bash

    autopep8 -i -a filename.py
    
For other problems you'll need to fix things by hand.  We aim to have no errors coming from the `pep8` program.


Testing
-------

We strive to test all code in bornagain, since we obviously want it to work in various environments rather than having
a quick hack that only works briefly, in one environment.

We use pytest to test the bornagain codebase.  It is simple to make a new test.  Just create a file
that has a name that beginning with `test_`, and within this file write functions that begin with `test_`, and within
those functions you include assert() statements.  These functions go into the test directory.  We then run pytest from
within the test directory, and all tests will be ran.

This isn't so hard, but it takes a lot of time.  Perhaps 25% of the code you write should be tests.  It is time
well-spent.


Generation of documentation
---------------------------

Docstrings from within the python code automatically find their way into this documentation via Sphinx.  Please keep
the formatting consistent by adhering to the Google-style doc strings.

If you modify code and wish to update this documentation, the easiest way to do so is to run the script
"update-documentation.sh" from within the doc directory.  First make sure you have all of the appropriate dependencies,
because sphinx must be able to import all of the bornagain modules in order to auto-generate module/package
documentation.

Speeding up code with Numba
---------------------------

Numba is one way to speed up Python code.  The basic idea is that you can write simple Python code and have it be
compiled on the fly so that it runs with speeds comparable to a language such as c.  We have used Numba in a couple
places, with some success, but there have been many instances when Numba has failed to compile code.  Last time I
tried, Numba did not have some basic methods for allocating arrays, such as numpy.zeros().  It seems ok to include
Numba code in bornagain, whenever it works, especially given that there have not yet been any issues with installing
Numba via the conda package manager.

Integration of Fortran and Numpy
--------------------------------

The f2py utility included with numpy is a convenient way to integrate fast CPU code
with numpy.  However, there are some issues with regard to passing numpy array pointers into a fortran function.
Fortran stores data in the so-called
"`column-major <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_" format, which means that an
increment in the left-most index of a multi-dimensional array corresponds to the smallest increment in the contiguous
block of memory.  In contrast, Numpy arrays are "row-major" by default, which means the right-most index corresonds
to contiguous increment.  This would not be a major problem if it weren't for the fact that Numpy arrays can also
be column-major, and it is hard to be sure of which indexing an
array might be using without explicitly checking the attributes such as "ndarray.flags.c_contiguous" (True if the
array is row-major) or "ndarray.flags.f_contiguous" (True if the array is column-major).  In Numpy language,
"C-contiguous" corresponds to row major, and "F-contiguous" corresponds to column major.

The above indexing syntax becomes important when you want to pass a Numpy array into a Fortran function with the
intention of actually modifying the Numpy array data.  When you use the f2py utility (see examples in
developer/compile-fortran.sh) to compile your Fortran code you will get runtime errors if you do not pass F-contiguous
arrays as inputs to the Fortran functions.  There is an exception: for
one-dimensional arrays, there is no distinction between F/C contiguous and no errors are raised.  There
is a convenience function "numpy.asfortranarray()" that will convert an array to a Fortran array, but
this is a very dubious function because it will make copies of the data, but it might not make the copy immediately and
this can potentially create a lot of confusion.

A good way to go is the following:

(1) Make sure that you always work with C-contiguous arrays in Python.  It makes no sense to work with a non-default
memory ordering.

(2) When you pass an array into a Fortran function, take the transpose of the array.  This will make the array
F-contiguous but will not make a copy of the memory.  Do not use the asfortranarray function for this purpose.

(3) In your Fortran code, simply reverse the ordering of indices as compared to your Numpy code.  You then use Fortran
reasoning in the Fortran code, and you assume the defaults in your Numpy code.

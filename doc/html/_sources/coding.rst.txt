.. _developers_anchor:

Notes for Developers
====================

Before you modify any code:
---------------------------

* The "`Zen of Python <https://www.python.org/dev/peps/pep-0020/>`_" captures the essence of Python programming.
* Follow the `PEP8 guidelines <https://www.python.org/dev/peps/pep-0008/?>`_.
* Use four spaces, not tabs.
* One exception to PEP8: we allow lines to be 120 characters in length.
* Write `unit tests <http://doc.pytest.org/>`_  for any functionality you add.
* *Always* write docstrings, and follow the `Google format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_.
* Learn how to use `git <https://git-scm.com/book/en/v2>`_.
* All units are SI (angles in radians) unless there is a *very* good reason to do something different.
* The scope of this project is diffraction under the Born approximation.  Don't stray far from this.

Python 2/3 compatibility
------------------------

To ensure compatibility with both Python 2 and 3, you may need to include something like the following at the beginning
of your module:

.. code-block:: python

    from __future__ import (absolute_import, division, print_function, unicode_literals)

Checking for PEP8 compliance
----------------------------

We use `pylint <https://www.pylint.org/>`_ for code formatting.  You should check that your code conforms to standards
as follows:

.. code-block:: bash

    pylint --max-line-length=120 filename.py

You can additionally use the pep8 program to check for inconsistencies (install pep8 with pip if need be).  In the
base directory of the git repo, do this

.. code-block:: bash

    pep8 bornagain
    
For simple errors like whitespace, you can use autopep8:

.. code-block:: bash

    autopep8 -i -a filename.py
    
For other problems you'll need to fix things by hand.  We aim to have no errors coming from the `pep8` program.


Testing
-------

We use `pytest <http://doc.pytest.org/>`_ to test the bornagain codebase.  It is very simple to make a new test.
Just create a file that has a name that beginning with `test_`, and within this file write functions that begin with
`test_`, and within those functions you include assert() statements.  These functions go into the test directory.  We
then run pytest from within the test directory, and all tests will be ran.


Generation of documentation
---------------------------

Docstrings from within the python code automatically find their way into this documentation via
`Sphinx <http://www.sphinx-doc.org/en/master/>`_.  Please keep
the formatting consistent by adhering to the Google-style doc strings.

If you modify code and wish to update this documentation, the easiest way to do so is to run the script
"update-documentation.sh" from within the doc directory.  First make sure you have all of the appropriate dependencies,
because sphinx must be able to import all of the bornagain modules in order to auto-generate module/package
documentation.

Speeding up code with Numba
---------------------------

Numba is one way to speed up Python code in cases where there is not an existing numpy function.  It is used within
bornagian in a few places and appears to be reasonably stable, though still lacking some basic functionality.

Integration of Fortran and Numpy
--------------------------------

The f2py utility included with numpy makes it quite easy to integrate simple Fortran code with Numpy.  However, there
are some issues that can arise when passing numpy array pointers into a fortran function.  This is especially
problematic when you want to use a Fortran routine to modify a memory buffer.  The root of the issues are due to the
complicated ways in which numpy ndarrays maintain their underlying memory buffers and the ways in which indexing is
used to access that memory at the Python level.  Examples of such complications can be found
in the `test_numpy.py` unit test.  It appears that the following recipe can be used to avoid any possible memory
issues

(1) Always work with the default C-contiguous ndarray memory layout in Python code.

(2) Use assert statements in function wrappers: `assert a.flags.c_contiguous == True`.

(3) Transpose ndarrays before passing them into Fortran functions.  This will *not* make a memory copy.

(4) In your Fortran code, simply reverse the ordering of your indices as compared to your Numpy code.

Although it may be inconvenient to reverse your indexing when going between the Fortran and Python code, bear in mind
that this can only be avoided by (a) making copies of array memory, or (b) using a non-default memory layout for Numpy
arrays.  Both options (a) and (b) are very bad.
.. _developers_anchor:

Developers
==========

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

    pep8 reborn
    
For simple errors like whitespace, you can use autopep8:

.. code-block:: bash

    autopep8 -i -a filename.py
    
For other problems you'll need to fix things by hand.  We aim to have no errors coming from the `pep8` program.


Testing
-------

We use `pytest <http://doc.pytest.org/>`_ to test the reborn codebase.  It is very simple to make a new test.

1) Create a file that has a name that beginning with `test_` in the reborn/test directory
2) Within this file, write functions with names that begin with `test_`
3) Within those functions, include assert statements.
4) Run pytest in the test directory, and all tests will run.


Generation of documentation
---------------------------

Docstrings from within the python code automatically find their way into this documentation via
`Sphinx <http://www.sphinx-doc.org/en/master/>`_.  Please keep
the formatting consistent by adhering to the Google-style doc strings.

If you modify code and wish to update this documentation, the easiest way to do so is to run the script
"update-documentation.sh" from within the doc directory.  First make sure you have all of the appropriate dependencies,
because sphinx must be able to import all of the reborn modules in order to auto-generate module/package
documentation.

Speeding up code with numba and f2py
------------------------------------

Numba is one way to speed up Python code in cases where there is not an existing numpy function.  It is used within
bornagian in a few places and appears to be reasonably stable, though still lacking some basic functionality.

A better way to speed up code is to use fortran.  There are notes on this elsewere in this documentation.


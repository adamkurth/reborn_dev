Coding guidelines
=================

First and foremost, the bornagain package does not presently follow the guidelines below, but we are presently trying to
fix the many inconsistencies.  Please contribute to this effort!!!

Before you modify any code:
---------------------------

* Read the "`Zen of Python <https://www.python.org/dev/peps/pep-0020/>`_".
* Follow the `PEP8 guidelines <https://www.python.org/dev/peps/pep-0008/?>`_.
* One exception to PEP8: we allow lines to be 120 characters in length.
* Strive to write `pytest <http://doc.pytest.org/>`_ unit tests for any functionality you add.
* *Always* write docstrings.  Follow the `Google format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_.
* Learn how to use git, and then read the bornagain-specific notes on git below.
* Use four spaces, not tabs.
* All units are SI (angles in radians) unless there is a *very* good reason to do something different.
* Just because you *can* do something clever, doesn't mean you should.  Don't confuse people.
* The scope of this project is diffraction under the Born approximation.  Don't stray far from this.

Helpful tools
-------------

* Look in the developer directory for helpful scripts such as "update-documentation.sh", which does what you guessed.
* Programs such as `pep8 <https://pypi.python.org/pypi/pep8/>`_ can be used to check for PEP8 compliance.
* Consider using an integrated development environment such as pycharm.

Using git
---------

* We will follow the branching scheme discussed here.
* Make your own branch to develop your ideas.
* When your branch is ready to be tested by others, first merge the develop branch into your branch, check that
  everything works (using pytest), then merge your branch into develop.
* When the develop branch seems to be working well, it will be merged into the master branch.

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

Outstanding issues
------------------

* We need a better way to manage documentation.  Ideally it would be available on the web, but I don't
  want it to be
  public until the API is reasonably stable.  At presently, nobody seems to read the documentation.
* We need a standardized way to test bornagain in the various combinations of Python 2, Python 3, pyqt4,
  pyqt5.  This may be possible with GitLab or GitHub.
* There are some classes, functions, methods that don't adhere to PEP8.  They will be changed, and old
  scripts will be
  broken.
* We need a good way to present example scripts in the documentation, which also shows output such as images.  The
  pyqtgraph package has a neat GUI demo when you do pyqtgraph.examples.run(), which we might emulate.
* We must revisit the way that we define vectors and rotation operations, and check for consistency
  throughout
* There are many more issues that need to be discussed.

Coding guidlines for developers
===============================

Overview of bornagain
---------------------

(This project is still in an early stage, so many of the guidelines below have not been followed.  We'll fix that in the next months.)

Some things to consider before writing code:

* The bornagain API should be thought of as an "interface" for analysis and simulation.  That is, we envision that the user will be working from an iPython prompt or similar, and they plan to accomplish something that can't be done easily with existing programs.
* The scope of this project is diffraction under the Born approximation.  It will not grow beyond that.
* Usability is more important than speed.  If speed is needed, we will tuck the details away (as in the clcore.py module).
* Always write tests along with modules.  We will use `pytest <http://doc.pytest.org/>`_ for testing.  Read about pytest.
* All units are SI (angles in radians) unless there is a *very* good reason to do something different.
* Follow the PEP8 guidelines to the greatest extent possible.  The `pep8 <https://pypi.python.org/pypi/pep8/>`_ program can be used to check for this.
* Use four spaces, not tabs.
* Just because you *can* do something clever, doesn't mean you should.
* Docstrings follow the `Google format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_.
* Never create a function without a docstring.  At least note what the function does.

Checking for PEP8 compliance
----------------------------

Use the `pep8` program to check for inconsistencies (install `pep8` with pip if need be).  In the base directory of the git repo, do this

.. code-block:: bash

    pep8 bornagain
    
For simple errors like whitespace, you can use `autopep8`:

.. code-block:: bash

    autopep8 -i -a filename.py
    
For other problems you'll need to fix things by hand.  We aim to have no errors coming from the `pep8` program.

Generation of documentation
---------------------------

So far we are following the Google-style doc strings.  There are some examples `here <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

Docstrings from within the python code find their way into this documentation via Sphinx.  If docstrings are changed, then the documentation should be updated.

If you modify code and wish to update this documentation, the easiest way to do so is to run the script "update-documentation" from within the doc directory.  First make sure you have the appropriate dependencies, as discussed in the installation section.

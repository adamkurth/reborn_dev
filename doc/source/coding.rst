Coding guidlines for developers
===============================

* The bornagain API is meant to be an "interface" for analysis and simulation.  That is, we envision that the user will be working from an iPython prompt or similar.
* Don't worry about speed until you need to worry about speed.  The interface is more important than speed.
* All units are SI (angles in radians) unless there is a very good reason to do something different.  Perhaps one needs to use degrees to specify an angle of exactly 90 degrees, for example.  The only time units should not be SI is when interfacing with other modules outside of bornagain.
* Follow the PEP8 guidelines to the greatest extent possible.
* Don't using tabs for indentation.  Use four spaces.
* Keep the code readable unless there is an extremely good reason to do otherwise.  Just because you *can* do something clever, doesn't mean you should...
  
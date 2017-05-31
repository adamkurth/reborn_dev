Overview
========

Here you will find an overview of basic concepts, which hopefully help you understand how the modules in bornagain are meant to fit together.

Some basics
-----------

- All units in bornagain are SI.  Angles are radians.  Exceptions are very rare.
- All vectors should have a numpy shape of Nx3, in order to keep vector components close in memory.  We are quite strict about this.
- Detector panels are often segmented in real experiments.  The classes in the detector module are intended to help ease the burden of working with such detectors.
- Learn how to use iPython to explore the functionality of bornagain classes.
- If documentation is missing, tell someone (e.g. rkirian at asu dot edu).

Detectors
---------

Some description about detectors here, with diagrams and usage examples.

Simulation
----------

Some description of how we work with opencl.
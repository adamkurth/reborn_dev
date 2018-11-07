Simulations
===========

TODO: Document the simulation code...

Currently, all GPU simulation utilities in bornagain utilize the pyopencl package.  You can check if opencl is working
and find out information with the :func:`help() <bornagain.simulate.clcore.help>` function.

The basic simulation functions are accessed by creating an instance of the
:class:`ClCore <bornagain.simulate.clcore.ClCore>` class.  This class is meant to do some of the following

- maintain an opencl context and queue
- manage the compute group size
- manage the precision (double/single)
- make it easy for you to move data between GPU memory and RAM (using the context and queue)

The above model seems to work well when using a single GPU device, but note that it has not been designed to
make use of multiple GPU devices (this might be easy to do, but we've not had a need thus far).

Aside from taking care of some low-level management of how the GPU device is used, the
:class:`ClCore <bornagain.simulate.clcore.ClCore>` class is primarily meant to provide only the most basic building blocks for
simulations, in the form of simple functions.  :class:`ClCore <bornagain.simulate.clcore.ClCore>` it is not meant to
manage data arrays; it is the lowest-level class
in the simulate package, and the basic idea is that you should make a specialized subclass for any specialized memory management.
(At the moment, there are some methods within :class:`ClCore <bornagain.simulate.clcore.ClCore>` that manage data arrays,
but they will be depreciated and moved to subclasses soon.)

Most of the method available through the :class:`ClCore <bornagain.simulate.clcore.ClCore>` class are
dedicated to computing the sum

    :math:`F(\vec{q}) = \sum_n f_n(q)e^{i\vec{q}\cdot\vec{r}_n}`

There are a couple of variations on how you can do this basic computation.  The most flexible way to compute the sum is to provide the
array of :math:`\vec{q}` vectors as an input to the function, in which case you obviously get to choose how those
vectors are defined.  There are two other options that might help improve speed: one variant computes the
:math:`\vec{q}` vectors corresponding to a PAD on the GPU
rather than getting them from global memory, and another variant computes the :math:`\vec{q}` vectors on a regular 3D
grid.  In the case that you compute :math:`F(\vec{q})` on a 3D grid, there is a corresponding function that can perform
interpolations such that you can sample :math:`F(\vec{q})` at any :math:`\vec{q}` that lies within the grid.

The methods in :class:`ClCore <bornagain.simulate.clcore.ClCore>` allow you to pass in CPU arrays and retrieve CPU
arrays in return.  Alternatively, you might want to do faster computations by copying/creating some arrays on the GPU device
ahead of the function call.  This alternative is especially useful if you plan to use the same array of :math:`\vec{q}`
vectors many times over.  You can move a numpy array to the GPU device with the :func:`to_device` method, and you can
also move the data from the GPU to the CPU with the :func:`get` method.

At a much higher level, there is a Monte Carlo simulator that generates crystal diffraction patterns.  This will be
documented by Rick and Cameron some day.
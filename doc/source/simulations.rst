Simulations
===========

TODO: Document the simulation code...

Currently, all GPU simulation utilities in bornagain utilize the pyopencl package.  You can check if opencl is installed correctly 
and find specific information about available compute devices with the :func:`help() <bornagain.simulate.clcore.help>` function.  
In principle, the pyopencl package also allows the use of CPU computing, but
none of the code has been written with that in mind thus far.  If it is needed
we can explore that option. 

The basic simulation functions are accessed by creating an instance of the
:class:`ClCore <bornagain.simulate.clcore.ClCore>` class.  This class is meant to do some of the following

- maintain an opencl context and queue
- manage the compute group size
- manage the precision (double/single)
- make it easy for you to move data between GPU memory and RAM (using the context and queue)

The above model seems to work well when using a single GPU device, but note that it has not been designed to
make use of multiple GPU devices (this might be easy to do, but we've not had a need thus far).

Aside from managing how the GPU device is used, the
:class:`ClCore <bornagain.simulate.clcore.ClCore>` class is meant to provide only the basic building blocks for
simulations, in the form of simple functions.  :class:`ClCore <bornagain.simulate.clcore.ClCore>` it is not meant to
manage data arrays; you should make a specialized subclass for any specialized memory management.  (At the moment, there
are some methods within :class:`ClCore <bornagain.simulate.clcore.ClCore>` that manage data arrays,
but they will be depreciated and moved to subclasses.)

Most of the methods in the :class:`ClCore <bornagain.simulate.clcore.ClCore>` class are dedicated to computing the sum

    :math:`F(\vec{q}) = \sum_n f_n(q)e^{i\vec{q}\cdot\vec{r}_n}`

Assuming that you have an array of atomic positions :math:`\vec{r}`, there are a couple of variations on how you can do
this basic computation.  The most flexible way to compute the sum is to explicitly provide an
array of :math:`\vec{q}` vectors as an input to the function, in which case you get to choose how those
vectors are defined.  There are two other options that might help improve speed: one variant computes the
:math:`\vec{q}` vectors corresponding to a PAD on the GPU
rather than getting them from global memory, and another variant computes the :math:`\vec{q}` vectors on a regular 3D
grid.  In the case that you compute :math:`F(\vec{q})` on a 3D grid, there is a corresponding function that can perform
interpolations such that you can sample :math:`F(\vec{q})` at any :math:`\vec{q}` that lies within the grid.

The methods in :class:`ClCore <bornagain.simulate.clcore.ClCore>` allow you to pass in CPU arrays and retrieve CPU
arrays in return.  
That is the simplest way to to GPU compuations since you can just use the methods without ever
thinking about memory.  However, the bottleneck in GPU computations is often due to moving memory between devices.
You can reduce computation time by copying/creating some or all of your arrays to the GPU device
ahead of the function call.  Likewise, you can retrieve the output in the form of pyopencl Array objects, which contain your data in GPU memory.  This alternative is especially useful if you plan to use the same array of :math:`\vec{q}`
vectors many times over - you obviously don't want to move your vectors between CPU and GPU many times over.  
Moving a numpy array to the GPU device is easy with the :func:`to_device` method, which is just a wrapper for the equivalent in the pyopencl package.  You can easily move the data from the GPU to the CPU with the :func:`get` method (this is just a built-in method of the pyopencl Array class).

At a much higher level, there is a Monte Carlo simulator that generates crystal diffraction patterns.  This will be
documented by Rick and Cameron some day.  The basic idea is that we have an engine that serves up diffraction intensities
in a Monte Carlo fashion, by jittering the incident beam direction, pixel positions, crystal orientations, and so on.

Other simulation engines might be built soon.  For example, an engine for solution scattering will come along soon.

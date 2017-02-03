# bornagain?

The name derives from the fact that this module is all about handling diffraction under the Born approximation, and of course this is not the first attempt of this sort (so Born... again...).  The previous module was named "pydiffract" (pydiffract has been born again).  

The aim here is to allow for flexible simulations and analysis of diffraction data.  Emphasis is placed on useability of the code for scripting tasks.  Some effort will also be dedicated to the creation of a decent diffraction viewer.  Development will be on an "as-needed" basis, but we will try to develop with some foresight of what might come next.

# Conventions

* All units are SI, and angles are radians.
* All vectors should have a numpy shape of Nx3.

# Dependencies:

In addition to the standard modules included with Anaconda Python 2.7:

`spglib` for space group symmetry:
```bash
pip install spglib
```

`pyqtgraph` for viewing:
```bash
conda install -c ufechner pyqtgraph=0.9.10
```

`pyopencl` for GPU simulations:
```bash
conda install -c timrudge pyopencl=2014.1
```

`xraylib` for simulations:
```bash
conda install -c conda-forge xraylib=3.2.0
```

`pyopengl` for plotting 3d graphics:
```bash
conda install pyopengl
```

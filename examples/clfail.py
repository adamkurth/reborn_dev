import pyopencl as cl
import pyopencl.clmath as clmath
import numpy as np

c = cl.create_some_context()
q = cl.CommandQueue(c)

a = np.zeros((1000,100),dtype=np.complex64)
a = cl.array.to_device(q,a)

b = np.ones((1000,100),dtype=np.complex64)
b = cl.array.to_device(q,b)

print(np.prod(b.shape))

c = a + b

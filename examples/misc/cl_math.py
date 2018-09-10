import pyopencl as cl
import pyopencl.clmath
import numpy as np

# Even though we don't call clmath explicitly, it seems to be necessary...
# Are there some hidden global variables lurking about?

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
c = cl.Context(devices=my_gpu_devices)
q = cl.CommandQueue(c)

a = np.random.random((5, 5)).astype(np.complex64)
a_gpu = cl.array.to_device(q, a)

b = np.random.random((5, 5)).astype(np.complex64)
b_gpu = cl.array.to_device(q, b)

# Test addition
c = a + b
c_gpu = a_gpu + b_gpu
c_gpu = c_gpu.get()
print(np.max(np.abs(c - c_gpu)))

# Test multiplication
c = a * b
c_gpu = a_gpu * b_gpu
c_gpu = c_gpu.get()
print(np.max(np.abs(c - c_gpu)))

# Test exponentiation
c = a**2
c_gpu = a_gpu**2
c_gpu = c_gpu.get()
print(np.max(np.abs(c - c_gpu)))
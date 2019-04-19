from time import time
import numpy as np


vecs = np.random.rand(100000, 3)
rot = np.random.rand(3, 3)
assert vecs.flags.c_contiguous
t = time()
for i in range(1000):
    rvec = np.dot(vecs, rot.T)
print('np.dot(vecs, rot.T)             : %g seconds' % (time()-t,))
assert rvec.flags.c_contiguous

vecs = np.random.rand(100000, 3)
rot = np.random.rand(3, 3)
assert vecs.flags.c_contiguous
t = time()
for i in range(1000):
    rvec = np.dot(vecs, rot.T.copy())
print('np.dot(vecs, rot.T.copy())      : %g seconds' % (time()-t,))
assert rvec.flags.c_contiguous

vecs = np.random.rand(100000, 3)
rot = np.random.rand(3, 3)
assert vecs.flags.c_contiguous
t = time()
for i in range(1000):
    rvec = np.dot(rot, vecs.T).T
print('np.dot(rot, vecs.T).T           : %g seconds' % (time()-t,))
assert rvec.flags.c_contiguous








def rotate(rot, vecs):
    return np.dot(vecs, rot.T)
vecs = np.random.rand(100000, 3)
rot = np.random.rand(3, 3)
assert vecs.flags.c_contiguous
t = time()
for i in range(1000):
    rvec = rotate(rot, vecs)
print('func np.dot(vecs, rot.T)        : %g seconds' % (time()-t,))
assert rvec.flags.c_contiguous

def rotate(rot, vecs):
    return np.dot(vecs, rot.T.copy())
vecs = np.random.rand(100000, 3)
rot = np.random.rand(3, 3)
assert vecs.flags.c_contiguous
t = time()
for i in range(1000):
    rvec = rotate(rot, vecs)
print('func np.dot(vecs, rot.T.copy()) : %g seconds' % (time()-t,))
assert rvec.flags.c_contiguous

def rotate(rot, vecs):
    return np.dot(rot, vecs.T).T
vecs = np.random.rand(100000, 3)
rot = np.random.rand(3, 3)
assert vecs.flags.c_contiguous
t = time()
for i in range(1000):
    rvec = rotate(rot, vecs)
print('func np.dot(rot, vecs.T).T      : %g seconds' % (time()-t,))
assert rvec.flags.c_contiguous


import sys
sys.path.append("..")
import numpy as np

from bornagain import utils

N = 5

# This is a properly shaped array list
a = np.zeros([N,3])
print(a.shape)
print(a.flags)

a = utils.vecCheck(a)
print(a.shape)
print(a.flags)

# This is a *improperly* shaped array list
a = np.zeros([3,N])
print(a.shape)
print(a.flags)

a = utils.vecCheck(a)
print(a.shape)
print(a.flags)

# This is an improperly formatted vector
t = np.arange(1,4)
print(t.shape)
print(t.flags)

t = utils.vecCheck(t)
print(t.shape)
print(t.flags)

# Broadcasting an overall translation
print(a + t)

b = np.arange(1,N+1)
b = np.reshape(b,[N,1])
print(b.shape)
print(a + b)
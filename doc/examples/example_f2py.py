r"""
.. _plot_f2py:

Mixing fortran with numpy using f2py
====================================

For advanced users: how to use f2py to write fortran code that operates on numpy arrays.

Contributed by Richard A. Kirian

First we import a function that has been generated by f2py:
"""

import numpy as np
from reborn.fortran.fortran_indexing_f import fortran_indexing

# %%
# Here is an important quote from the f2py docs:
#
#   *In general, if a NumPy array is proper-contiguous and has a proper type then it is directly passed to wrapped
#   Fortran/C function. Otherwise, an element-wise copy of an input array is made and the copy, being
#   proper-contiguous and with proper type, is used as an array argument.*
#
# The above statement explains why a lot of confusion can arise when passing arrays to wrapped fortran functions:
# copies of data are made silently, depending upon how they are formatted.  This results in inconsistent behavior as
# perceived by someone who hasn't fully read the docs.
#
# In order to help understand the problem better, we provide a simple fortran test function.  The native fortran
# function "fortran_indexing" does the following (verbatim, from the source code):
#
# .. code:: fortran
#
#     subroutine fortran_indexing(out1, out2, out3)
#     implicit none
#     real(kind=8), intent(inout):: out1(:), out2(:,:), out3(:,:,:)
#     out1(2) = 10
#     out2(2, 1) = 10
#     out3(2, 1, 1) = 10
#     end subroutine fortran_indexing
#
# The wrapper function generated by f2py aims to achieve the following:
#
# .. code:: python
#
#     def fortran_indexing(out1, out2, out3)
#         out1[1] = 10
#         out2[1,0] = 10
#         out3[1,0,0] = 10
#
# What's wrong with the above?  Answer: the fortran code accesses array values that are second in memory, but the
# equivalent numpy function appears to access elements that are *not* the second in memory. This is because the default
# numpy array is c-contiguous, in which case the right-most index increments along adjacent elements in memory, whereas
# fortran syntax is reversed: it is f-contiguous and the left-most index increments along adjacent elements in memory.
# Here are some ways to avoid this issue:
#
# * Method 1: Make sure that all of your numpy arrays are f-contiguous, rather than the default c-contiguous.  This is
#   both annoying and likely to slow down other numpy code that assumes default contiguity.
#
# * Method 2: Write additional function wrappers that make f-contiguous copies of your arbitrary-contiguous numpy
#   arrays.  This wastes time and memory.
#
# * Method 3: Pass the default c-contiguous numpy memory buffers directly to the fortran function, and use reversed
#   index ordering when comparing fortran code against the equivalent numpy code.
#
# We identify Method 3 as the best approach, because this method will have you think about fortran coding in the most
# natural way, and you will also think about numpy coding in the most natural way.  This minimizes the likelihood of
# making copies of arrays or having slow execution.
#
# Before we get started, let's find out why it is a bad idea to simply wrap your arrays using the
# ``numpy.asfortranarray`` function, as suggested in some examples on the web.  Start with default numpy arrays:
out1 = np.zeros(10)
out2 = np.zeros((10, 10))
out3 = np.zeros((10, 10, 10))
# %%
# Pop quiz: are the above three arrays (a) c-contiguous or (b) f-contiguous?
fortran_indexing(np.asfortranarray(out1), np.asfortranarray(out2), np.asfortranarray(out3))
# %%
# Note that the first array, ``out1``, is both f-contiguous __AND__ c-contiguous.  Why?  Because it is one dimensional!!
assert out1.flags.f_contiguous
assert out2.flags.c_contiguous
# %%
# In the case of 1D arrays, the fortran function wrapper will never (?) make copies of arrays.  That means that your
# fortran code will directly operate on the input array.  Let's show that the input array has been modified, which
# may or may not be your intention:
assert out1[1] == 10
# %%
# Now, ``out2`` is *not* f-contiguous.  It is the default c-contiguous order.  Why?  Because it is a 2D array.  The
# notion of c/f contiguity is relevant only to multi-dimensional arrays!
assert out2.flags.c_contiguous
assert not out2.flags.f_contiguous
# %%
# Because ``out2`` was not f-contiguous, it seems that the fortran function wrapper made an f-contiguous copy.  As seen
# below, the original ``out2`` array has *not* been modified by the fortran function.  Conclusion so far?  You better
# be mindful of both contiguity *and* dimension of your arrays if you want consistent behavior.
assert out2[1, 0] != 10
# %%
# The behavior of 2D arrays is similar to 3+D arrays:
assert out3.flags.c_contiguous
assert out3[1, 0, 0] != 10
# %%
# Let's do things slightly differently.  We'll make a new variable with asfortranarray, ``out2``, which differs from the
# initial c-contiguous array ``out2_0``.
out1 = np.zeros(10)
out2_0 = np.zeros((10, 10))
out2 = np.asfortranarray(out2_0)
# %%
# What do you suppose asfortranarray really does?
assert out2_0.flags.c_contiguous
assert not out2_0.flags.f_contiguous
assert not out2.flags.c_contiguous
assert out2.flags.f_contiguous
# %%
# Look above: one is c-contiguous, the other is f-contiguous.  Get ready to be very confused...
assert out2_0.data == out2.data
# %%
# From the above, both arrays share the *same data buffer*, but we are also told that one is c-contiguous while the
# other is f-contiguous.  How can this be?
out3 = np.zeros((10, 10, 10))
fortran_indexing(out1, out2, np.asfortranarray(out3))
assert out1.flags.f_contiguous
assert out1[1] == 10
assert out2.flags.f_contiguous
assert out2[1, 0] == 10
assert out2_0[1, 0] != 10
assert out2_0.data != out2.data
# %%
# Now we see that passing the ``out2`` array to the fortran_indexing function causes the underlying data
# buffer to change.  This is very confusing and unfortunate behavior.

# %%
# Here comes Method 3:
# Work with the native numpy ordering (c-contiguous).  Do not attempt to make your arrays f-contigous.
out1 = np.zeros(10)
out2 = np.zeros((10, 10))
out3 = np.zeros((10, 10, 10))
# %%
# Assert that all arrays going into a f2py-generated function are c-contiguous.
assert out1.flags.c_contiguous
assert out2.flags.c_contiguous
assert out3.flags.c_contiguous
# %%
# Now it's time to use the fortran function.  Please, don't even think about using asfortranarray.
# Here is a simple trick: pass the transposes of your c-contiguous arrays to the fortran function:
fortran_indexing(out1.T, out2.T, out3.T)
# %%
# The result of doing the above is that you won't make any copies of arrays, and you'll work using the normal
# way of thinking about indexing and contiguity when coding in both fortran and numpy.  Think about what the
# fortran function does -- it modifies the second element in memory.  Now, the numpy equivalent does the same thing:
# modifies the second element in memory.  Proof:
assert out1[1] == 10
assert out2[0, 1] == 10
assert out3[0, 0, 1] == 10
# %%
# Yes, the syntax in fortran looks different from numpy: the default fortran indexing (2,1,1) corresponds to the
# default of [0,0,1] in numpy.  But that's good: they look different because they *are* different!  Our goal is not to
# achieve matching indexing syntax in the two different languages.  The goal is to have functions that actually do the
# same thing to the expected memory buffers...
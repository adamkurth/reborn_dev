r"""
Making real space density maps
===============
"""

# %%
# Test


"""
# %%
# This is meant to have some notes that will help developers who intend to add examples to this gallery.
# We are using Sphinx Gallery, so you should look to that documentation for details.  Here we'll just include
# some very basic usage notes.
#
# Comments vs. documentation blocks
# ---------------------------------
#
# You need to start a comment block with '# %%' in order for it to show up in the style you are presently
# reading.  If you just use normal python comments (i.e. start a line with '#') then it will be formatted
# as a code block.  For more details, look here: https://sphinx-gallery.github.io/stable/syntax.html
#
# Basic usage
# -----------
#
# Importantly, if you want some kind of plot to show up in the docs, then you need to prefix the name of
# your file with "plot_".  Then, magically, plots will show up in the documentation!


import numpy as np
import matplotlib.pyplot as plt

a = np.arange(10)
plt.plot(a)
plt.show()

# %%
# You can include LaTeX equations like this:
#
# .. math::
#
#     \vec{q} = \frac{2\pi}{\lambda}(\vec{s}-\vec{s}_0)
#
# Here is some more code.  Notably, the imports from the previous block will still exist in the next block.
# Let's check:

b = a
print(b)

# %%
# The above code block also shows what happens when you add a print statement into a separate block of code.
"""
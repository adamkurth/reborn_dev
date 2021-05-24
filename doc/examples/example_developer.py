r"""
How to contribute an example
============================

Brief overview of how to contribute an example script.

Contributed by Richard Kirian.

"""

# %%
# We are using `Sphinx Gallery <https://sphinx-gallery.github.io/stable/index.html>`_ to auto-generate examples such as
# the one you are presently reading.  Each time a new commit is pushed to the master branch on gitlab, the example
# scripts will run under the coordination of Sphinx Gallery.  You should look to the official documentation for more
# detail on how Sphinx Gallery.
#
# In a nutshell, examples like this one should be placed in the reborn/doc/examples directory.  Any example that goes
# there should run in a reasonable amount of time (seconds, not minutes!) since there is a timeout on the (free)
# gitlab runners that we use to generate documentation.
#
# With the above understood, let's take a quick look at the anatomy of an example script.
#
# The first doc string
# --------------------
#
# Your file *must* have a triple-quote doc string at the beginning of the file, and there *must* be a very short
# descriptive title underlined with = signs.  Please also include a line: "Contributed by your name" or if you modify
# an example include the line "Modified by your name".


# %%
# Code comments vs. documentation blocks
# --------------------------------------
#
# If you want nicely formatted blocks of documentation text rather than monospace code comments in the 
# raw code, start your comment block with `# %%`.  If you do that, your text will appear in the style you are presently
# reading.  For more details, look here: https://sphinx-gallery.github.io/stable/syntax.html
#
# How to make the code run
# ------------------------
#
# Importantly, if you want your code to actually run and produce output, then you need to prefix the name of
# your file with `example_`.  Then, magically, examples will show up in the documentation!

import numpy as np
import matplotlib.pyplot as plt

a = np.arange(100)*2*np.pi/100
plt.plot(np.sin(a))
plt.title('A sin wave!')
plt.show()

# %%
# How to add equations
# --------------------
#
# In the above, the Sphinx Gallery somehow figures out how to capture images of the matplotlib displays.
# If you use a different package to display something, you probably won't get a plot to show up here.
#
# 
# You can include LaTeX equations like this:
#
# .. math::
#
#     \vec{q} = \frac{2\pi}{\lambda}(\vec{s}-\vec{s}_0)

# %%
# Other notes
# -----------
#
# Here is some more code.  Notably, the imports from the previous block will still exist in the next block.
# Let's check:

b = a
print(b)

# %%
# The above code block also shows what happens when you add a print statement into a separate block of code.

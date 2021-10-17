# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

r"""
.. _example_pad_geometry:

Working with Pixel Array Detectors (PADs)
=========================================

Demonstration of how to work with Pixel Array Detectors using the reborn tools.

Contributed by Richard Kirian.

Edited by Konstantinos Karpos

"""

# %%
# Overview
# --------
#
# Nearly everyone who works with XFEL diffraction gets confused when they encounter data that consists of multiple
# Pixel Array Detectors (PADs).  This happens when people get used to synchrotron detectors that consist of single
# PADs (usually fake single PADs created by software), or when students begin with simulation projects and later try to
# adapt their code to real data.  In this example, we hope to provide guidance on how to write code that handles
# multiple PADs.
#
# .. note::
#
#     This documentation does not describe what a PAD is.  If you are unfamiliar with the basic concept of a PAD and the
#     vectors used to describe PAD geometry in reborn, first read the :ref:`documentation <doc_pads>` before moving
#     forward from this point.
#
# The starting point for working with PAD data is to specify the geometry of the PADs.  The reborn package provides the
# |PADGeometry| class to do so in a *standardized way*.  Note that a single |PADGeometry|
# instance only contains the information of a *single* PAD; you need multiple |PADGeometry| instances to handle multiple
# PADs.  We will get to that later in this example.


# %%
# Working with a single PAD
# -------------------------
#
# There are various ways to create a |PADGeometry| instance.  The "manual" way is to create the instance without any
# kind of initialization:

import numpy as np
from scipy.ndimage import gaussian_filter
from reborn import detector, source, temp_dir
from reborn.viewers.mplviews import view_pad_data

pad = detector.PADGeometry()
print(pad)

# %%
# As you see, none of the five necessary parameters are specified.  We can specify them manually:

pad.n_fs = 100
pad.n_ss = 200
pad.fs_vec = [100e-6, 0, 0]
pad.ss_vec = [0, 100e-6, 0]

# %%
# In the above, we specified only four of the five necessary parameters.  As a result, your code will likely raise
# ValueErrors.  For convenience, you may validate your |PADGeometry|:

try:
    pad.validate()
except ValueError:
    print('ValueError excepted')

# %%
# Now we add in the final parameter, which is the overall translation of the PAD.  Be mindful of the fact that we will
# define "detector distance", and then set it to the third coordinate (i.e. "z") of the translation vector.  This of
# course depends on how you define "detector distance".  In our case, we understand that "detector distance" refers to
# the "z" component, which is the third component of our vectors, and which is orthogonal to the gravitational force
# (to be absolutely clear...).  It is not uncommon for others to assume the x-ray beam points in a totally different
# direction.  In fact, the x-ray beam rarely points exactly along the so-called "z" direction, because KB focusing
# optics deflect the beam (for example).  When defining PAD geometry parameters, one must think.

detector_distance = 1
pad.t_vec = [-50*100e-6, -100*100e-6, detector_distance]

# %%
# Although we provided an ordinary 3-element python list when defining this vector, the t_vec property definition in
# |PADGeometry| ensures that the internal vector is a properly-shaped |ndarray|, which we can confirm:

print(pad.t_vec.shape)

# %%
# Now that we know how to configure a |PADGeometry| "manually", we point out that there is an initialization function
# that you might find useful:

pad = detector.PADGeometry(distance=detector_distance, shape=(100, 200), pixel_size=100e-6)

# %%
# The above function assumes that the beam detector_distance refers to the "z" coordinate, and it also assumes that the
# x-ray beam is incident on the center of the PAD.
#
# Now that we have a |PADGeometry| set up, we can use it to get useful quantities.  For example, the solid angles of the
# pixels are

sa = pad.solid_angles()

# %%
# Importantly, the shape of the solid angle array is not 2D, as you may expect.  Let's check:

print(sa.shape)

# %%
# As we will see shortly, it is often most convenient to work with flattened arrays.
#
# You can look to the |PADGeometry| documentation to see other useful quantities.  For example, the vectors pointing
# from the origin to each of the pixels are

vecs = pad.position_vecs()
print(vecs.shape)

# %%
# Once again, the output shape of these vectors may not be what you expect -- they are in the standard format for
# specifying multiple vectors in reborn.

# %%
# One of the most frequently used quantities that |PADGeometry| provides are the q vectors.  However, a |PADGeometry|
# class alone cannot provide q vectors, because q vectors, by definition, require knowledge of the x-ray wavelength and
# the direction of the beam.  Here is one way to get the q vectors:

q_vecs = pad.q_vecs(beam_vec=[0, 0, 1], wavelength=1.5e-10)

# %%
# As in the above, we provided the beam direction and wavelength, which helps clarify what is meant when we say "reborn
# makes no prior assumptions about geometry".  Since beam direction and wavelength are frequently needed, reborn
# provides the |Beam| class as a compact container for various beam properties.  For example:

beam = source.Beam(wavelength=1.5e-10, pulse_energy=1e-3)
q_vecs = pad.q_vecs(beam=beam)  # noqa

# %%
# In the above, the default beam direction for the |Beam| class is [0, 0, 1], but you may configure this as you see fit.
#
# %%
# There are other quantities that require beam information, such as the polarization factors:

polfacs = pad.polarization_factors(beam=beam)

# %%
# As shown above, once you have a |Beam| instance, it is straightforward to pass that information to your
# |PADGeometry|.  It is advisable to use the |Beam| class if you are already using the |PADGeometry| class.
#
# Note that reborn provides functions to save and load |PADGeometry| information:

filename = temp_dir + 'pads.json'
detector.save_pad_geometry_list(filename, pad)
pad2 = detector.load_pad_geometry_list(filename)

# %%
# Now have a look at pad2:

print(type(pad2))

# %%
# As you can see, it is a |PADGeometryList|, which naturally brings us to the next section.

# %%
# Working with Multiple PADs
# --------------------------
#
# As noted previously, it is very common to have multiple PADs when working with XFEL data.  As such, much of the
# reborn package is built around the assumption that you may have more than one PAD.  One way to handle multiple PADs is
# to simply make a python list:

psize = 1e-3
pad1 = detector.PADGeometry(distance=0.1, shape=(100, 100), pixel_size=psize)
pad2 = detector.PADGeometry(distance=0.1, shape=(100, 100), pixel_size=psize)
pads = [pad1, pad2]

# %%
# In the above, both pad1 and pad2 have the same geometry, so we'll need to fix that so that they do not collide.  We
# shift the detectors, which were initially centered:

pad1.t_vec[0] -= 51*psize
pad2.t_vec[0] += 51*psize

# %%
# At this stage, it would be useful to be able to visualize our PAD list, so let's create some data to look at.  For
# convenience, we set our data equal to a totally uniform scatterer, and we include the polarization factor:

data = [np.random.random(p.n_pixels)*p.polarization_factors(beam=beam) for p in pads]
print(pads[0])

# %%
# The reborn package provides tools to view lists of data arrays, provided matching lists of |PADGeometry| instances:

view_pad_data(pad_data=data, pad_geometry=pads, pad_numbers=True)

# %%
# We now have a list of data arrays along with corresponding PAD geometry information.  We could write analysis code
# around these lists.  For example, suppose we wish to correct for the polarization factor.

for (p, d) in zip(pads, data):
    d /= p.polarization_factors(beam=beam)
view_pad_data(pad_data=data, pad_geometry=pads, pad_numbers=True)

# %%
# The above is in essence how one must handle multiple PADs.  Looping over multiple PADs is inescapable fact of life
# for folks who analyze XFEL data.  It is a nuisance, but reborn attempts to make the process less annoying.
#
# The |PADGeometryList| class in reborn is a sub-class of the built-in python list class, so it has all the
# features of a python list instance, but it adds a few methods that are specific to lists of |PADGeometry|.  This is
# perhaps easiest to explain by doing the same operations as the above:

pads = detector.PADGeometryList(pads)  # Transform normal list to PADGeometryList
data = np.random.random(pads.n_pixels)*pads.polarization_factors(beam=beam)
print(data.shape)

# %%
# As you can see, data is now a single |ndarray|, and it's length is equal to the combined length of all PADs in the
# list.  This is a concatenated data array, and it is particularly useful for array operations that benefit from
# vectorization, such as the simple multiplication we did above.  We can display the data as we did before:

view_pad_data(pad_data=data, pad_geometry=pads, pad_numbers=True)

# %%
# Note that, because the view_pad_data function was given a |PADGeometryList|, it was able to split up the data even
# though a concatenated array was passed in.  We try to enable such capabilities in reborn wherever it is appropriate,
# but it is usually best if you split up your data into individual panels once you are done with your array-based
# operations.  This is easy to do:

data_split = pads.split_data(data)
print(type(data_split))
# %%
print(data_split[0].shape)
# %%
data_concat = pads.concat_data(data_split)
print(data_concat.shape)

# %%
# As shown above, we can easily move between concatenated data arrays and lists of individual 2D |ndarray| s.  If you
# are doing image-processing steps using functions in scipy (for example), you will almost certainly need to write loops
# over the individual 2D |ndarray| s.  For example:

for i in range(len(data_split)):
    data_split[i] = gaussian_filter(data_split[i], sigma=4)
view_pad_data(pad_data=data_split, pad_geometry=pads, pad_numbers=True)

# %%
# The |PADGeometryList| class attempts to provide all of the generalizations to the corresponding |PADGeometry| methods
# that you'd hope for:

if pads.validate():
    print('Good job.')
for p in pads:
    print(p.n_pixels)
print(pads.n_pixels)
q_vecs = pads.q_vecs(beam=beam)
print(q_vecs.shape)
q_mags = pads.q_mags(beam=beam)
print(q_mags.shape)
p_vecs = pads.position_vecs()
print(p_vecs.shape)

# %%
# Importing Pre-Built Detectors
# -----------------------------

# %%
# `reborn` provides multiple detector geometries that are ready for use. Say your experiment used the MPCCD 
# detector and you would like to display a 2D pattern. The `reborn.detector.mpccd_pad_geometry_list()` function 
# will provide that detector as a |PADGeometryList| object, which can be manipulated using the methods described above.

mpccd_geom = detector.mpccd_pad_geometry_list(detector_distance=0.05)
data = [np.random.random(p.n_pixels)*p.polarization_factors(beam=beam) for p in mpccd_geom]

view_pad_data(pad_geometry=mpccd_geom, pad_data=data)

# %%
# Note the use of multiple panels here. Changing panel properties is easy from here. As as example,
# say you would like to rotate your detector by 45 degrees. 

theta = 45 * np.pi / 180  # reborn uses radians by default

# define your rotation matrix
R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

# %%
# Let's see what this looks like with one panel, first.

geom_1 = mpccd_geom.copy()  # make a copy of the pad geometry so you don't mess with the original
geom_1[0].fs_vec = np.dot(geom_1[0].fs_vec, R.T)
geom_1[0].ss_vec = np.dot(geom_1[0].ss_vec, R.T)

view_pad_data(pad_geometry=geom_1, pad_data=data)

# %%
# Cool, so the bottom left panel was rotated by 45 degrees! Note that the `view_pad_data()`
# function is a projection operation, so although it looks like it simply shrunk, the panel 
# shows a 2D projected rotation.

# %%
# Now let's apply to this to the whole detector.
geom_2 = mpccd_geom.copy()  # make a copy of the pad geometry so you don't mess with the original

# Loop across each panel and perform the rotation operation
for pad in geom_2:
    pad.fs_vec = np.dot(pad.fs_vec, R.T)
    pad.ss_vec = np.dot(pad.ss_vec, R.T)
    pad.t_vec = np.dot(pad.t_vec, R.T)

# view the rotation
view_pad_data(pad_geometry=geom_2, pad_data=data)

# %%
# Dealing with Multiple Detectors
# -------------------------------

# %% 
# In some cases, you may need to work with more than one "detector". While most experiments use a single "detector"
# (note: a "detector" often consists of several identical PADs), there have been experiments that utilize multiple
# detectors to combine both high-resolution and low-resolution data. For this example, we'll use the Rayonix MX340 and
# the Epix 10K detectors. Note that all detector distances are in meters.

rayonix_geom = detector.rayonix_mx340_xfel_pad_geometry_list(detector_distance=0.5)
epix_geom = detector.epix10k_pad_geometry_list(detector_distance=0.1)

# Offset the epix detector to visualize them side by side

for pad in epix_geom:
    pad.t_vec[0] += 0.2  # translate the epix detector 20 cm to the right

# %%
# In order to display data across multiple panels, you will have to concatenate the two detectors. 
# You can do this easily with the following function,

c_geom = detector.PADGeometryList(rayonix_geom + epix_geom)

# %% 
# Note that from here on out, the order of the detectors matters. You will get an incorrect 2D
# pattern if you switch the order of the detectors. 

# Create some arbitrary data
data = [np.random.random(p.n_pixels)*p.polarization_factors(beam=beam) for p in c_geom]
view_pad_data(pad_geometry=c_geom, pad_data=data)

# %%
# Conclusions
# -----------

# %%
# The above should give you a pretty good start.  It is advisable that you go ahead and write your code such that it is
# generalized to an arbitrary number of PADs.  It is not very painful once you know how to make use of the reborn tools.
#
# If you think you need further examples, please let Rick know and he will add them here.  Similarly, if you feel that
# functionality is missing, request that it be added.  Most requests are added within a day.

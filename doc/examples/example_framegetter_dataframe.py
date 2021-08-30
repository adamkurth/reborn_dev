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
Working with the FrameGetter class
==================================

Brief overview of how to use reborn FrameGetters.

Contributed by Richard A. Kirian.
"""

# %
# In this example we cover a few of the basic concepts behind the reborn FrameGetter class, which is meant to help
# structure consistent workflows for simulations or data processing.

# %
# First a few imports.

import numpy as np
from reborn import detector, source, dataframe
from reborn.simulate import solutions
from reborn.viewers.qtviews import PADView
from reborn.fileio.getters import FrameGetter
np.random.seed(0)

# %
# For this example, we will make a FrameGetter that serves up simulations of water scatter.  The basic idea is that
# you need to *subclass* FrameGetter in order to make a useful FrameGetter.  If the notion of making a subclass
# is new to you, then you need to spend some time learning about object-oriented Python programming before moving on.
# Assuming that you understand the idea of subclassing, we will now make an example subclass:

class WaterFrameGetter(FrameGetter):
    def __init__(self, pad_geometry, beam, n_frames=10):
        super().__init__()  # Don't forget to initialize the base FrameGetter class.
        self.pad_geometry = pad_geometry
        self.beam = beam
        self.n_frames = n_frames  # This is important; you must configure the number of frames.
    def get_data(self, frame_number):
        intensity = solutions.get_pad_solution_intensity(pad_geometry=self.pad_geometry, beam=self.beam, liquid='water',
                                                         thickness=5e-6, temperature=298, poisson=True)
        df = dataframe.DataFrame()  # This creates a DataFrame instance, which combines PADGeometry with Beam and data.
        df.set_frame_index(self.current_frame)  # Not mandatory, but good practice.
        df.set_frame_id('Water %s' % frame_number)  # Not mandatory, but good practice.
        df.set_raw_data(intensity)
        df.set_pad_geometry(self.pad_geometry)
        df.set_beam(self.beam)
        df.validate()  # Not mandatory, but good practice.
        return df

# %
# As yo usee in the above example, there are just two basic steps:
#
# 1) Define the __init__ method.  Here you may store whatever data is needed in the subsequent calls to the get_data
#    method defined next.  Pay attention to the comments regarding n_frames and super().__init__().
# 2) Define the get_data method.  Here we add whatever code is needed to fetch data corresponding to a given
#    frame number.  It is assumed that you can index your data with integers, although this is something that we may
#    relax if the need arises.  Note that we pass back an instance of the DataFrame class, which is helpful because
#    other parts of reborn know what to do with DataFrame class instances.
#
# Now let's create an actual instance of our FrameGetter subclass:

pads = detector.cspad_2x2_pad_geometry_list(detector_distance=0.01).binned(4)
beam = source.Beam(photon_energy=6000*1.602e-19, diameter_fwhm=5e-6, pulse_energy=0.1e-3)
my_framegetter = WaterFrameGetter(pad_geometry=pads, beam=beam)

# %
# Now we give it a try.  We can access a particular frame:

df = my_framegetter.get_frame(2)
print(df.get_frame_index())

# %
# We can also request the next frame:

df = my_framegetter.get_next_frame()
print(df.get_frame_index())

# %
# We an loop through frames as we would with any other iterator:

for df in my_framegetter:
    print(df.get_frame_index())

# %
# We can get a random frame:

df = my_framegetter.get_random_frame()
print(df.get_frame_index())

# %
# The history of loaded frames is cached automatically, so that we can go back to the previous frame:

df = my_framegetter.get_random_frame()
print(df.get_frame_index())
df = my_framegetter.get_history_previous()
print(df.get_frame_index())
df = my_framegetter.get_history_previous()
print(df.get_frame_index())

# %
# We can also pass the FrameGetter to PADView in order to quickly view frames:

pv = PADView(frame_getter=my_framegetter)
pv.start()

# %
# With the above, you can flip through frames by pressing the left/right arrows, or the 'r' button to see a random
# frame, and so on (see the PADView example).

# %
# As you can see, FrameGetter does not do very much at this time aside from creating a "standard" interface for using
# PADView.  In the future, we will add to FrameGetter some capability to cache frames via multi-threaded reading, which
# will help reduce the time spent loading data from disk.  We will also accumulate some example FrameGetter subclasses
# (in addition to, for example, the StreamfileFrameGetter subclass found in external.crystfel).














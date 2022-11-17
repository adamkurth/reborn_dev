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

import numpy as np
from reborn import detector, source, dataframe, temp_dir
from reborn.analysis import runstats
from reborn.fileio.getters import ListFrameGetter


def test_padstats():
    geom = detector.cspad_2x2_pad_geometry_list()
    geom = geom.binned(10)
    beam = source.Beam(wavelength=1e-10)
    dataframes = []
    for i in range(3):
        dat = geom.zeros() + i
        df = dataframe.DataFrame(raw_data=dat, pad_geometry=geom, beam=beam)
        dataframes.append(df)
    fg = ListFrameGetter(dataframes)
    histparams = dict(bin_min=0, bin_max=10, n_bins=11, n_pixels=geom.n_pixels)
    stats = runstats.padstats(framegetter=fg, histogram_params=histparams)
    assert(isinstance(stats, dict))
    assert(stats['sum'].flat[0] == 3)
    # runstats.view_padstats(stats)
    filepath = temp_dir + '/stats.npz'
    runstats.save_padstats(stats, filepath)
    stats2 = runstats.load_padstats(filepath)
    assert(isinstance(stats, dict))
    assert(stats['sum'].flat[0] == stats2['sum'].flat[0])

import sys
import numpy as np
sys.path.append('..')
import bornagain as ba
from bornagain.utils import vec_check


def test_panel(main=False):

    ###########################################################################
    # Test the initial setup of a panel
    ###########################################################################

    nF = 3
    nS = 4
    pixel_size = 100e-6
    distance = 1.5
    wavelength = 1.5e-10
    T = (1.5, 0, distance)
    F = [1., 0, 0]
    S = (0, 1., 0)

    # Check all defaults
    p = ba.detector.Panel()
    assert(p.nF == 0)
    assert(p.nS == 0)
    assert(p.F is None)
    assert(p.S is None)
    assert(p.T is None)
    assert(p.name == "")
    assert(p.panellist is None)

    # It should be possible to use any sort of 3-tuple to set a vector
    p.T = T
    p.F = F
    p.S = S
    p.nF = nF
    p.nS = nS
    p.beam.wavelength = wavelength

    ###########################################################################
    # Check creation and clearing of geometry cache
    ###########################################################################

    # Check that cache is cleared
    def cache_cleared(p):
        assert(p._v is None)
        assert(p._sa is None)
        assert(p._pf is None)
        assert(p._k is None)
        assert(p._ps is None)
        assert(p._rsbb is None)
        assert(p._gh is None)
    # Create cache (by requesting the relevant arrays)

    def create_cache(p):
        V = p.V
        sa = p.solid_angle
        pf = p.polarization_factor
        q = p.Q
        ps = p.pixel_size
        rsbb = p.real_space_bounding_box
        gh = p.geometry_hash
    create_cache(p)
    p.T = T  # This should clear the cache
    cache_cleared(p)
    create_cache(p)
    p.F = F  # This should clear the cache
    cache_cleared(p)
    create_cache(p)
    p.S = S  # This should clear the cache
    cache_cleared(p)
    create_cache(p)
    p.pixel_size = pixel_size  # This should clear the cache
    cache_cleared(p)
    create_cache(p)
    p.nF = nF  # This should clear the cache
    cache_cleared(p)
    create_cache(p)
    p.nS = nS  # This should clear the cache
    cache_cleared(p)

    ###########################################################################
    # Test simple setup utility
    ###########################################################################

    p.simple_setup(nF=nF, nS=nS, pixel_size=1,
                   distance=distance, wavelength=wavelength)
    assert(p.V[0, 0] == -nF / 2.0 + 0.5)
    assert(p.V[-1, 0] == nF / 2.0 - 0.5)
    assert(p.V[0, 1] == -nS / 2.0 + 0.5)
    assert(p.V[-1, 1] == nS / 2.0 - 0.5)

    ###########################################################################
    # Test that geometry hash works
    ###########################################################################

    def panel_setup():
        p = ba.detector.Panel()
        p.simple_setup(nF=3, nS=4, pixel_size=1, distance=1, wavelength=1)
        return p

    # Geometry hash is None by default
    p = ba.detector.Panel()
    assert(p.geometry_hash is None)

    # Self consistency
    p = panel_setup()
    assert(p.geometry_hash is not None)
    assert(p.geometry_hash == p.geometry_hash)

    # Comparison between two identical setups, different objects
    p1 = panel_setup()
    p2 = panel_setup()
    assert(p1.geometry_hash == p2.geometry_hash)

    # Check mismatch if something is changed
    p1 = panel_setup()
    p2 = panel_setup()
    p2.nF += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panel_setup()
    p2 = panel_setup()
    p2.nS += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panel_setup()
    p2 = panel_setup()
    p2.F += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panel_setup()
    p2 = panel_setup()
    p2.S += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panel_setup()
    p2 = panel_setup()
    p2.T += 1
    assert(p1.geometry_hash != p2.geometry_hash)


def test_panellist(main=False):

    def panel1_setup():
        p = ba.detector.Panel()
        p.simple_setup(nF=3, nS=4, pixel_size=1, distance=1, wavelength=1)
        return p

    def panel2_setup():
        p = ba.detector.Panel()
        p.simple_setup(nF=4, nS=5, pixel_size=1, distance=1, wavelength=1)
        return p

    ###########################################################################
    # Test the initial setup of a panellist
    ###########################################################################

    pl = ba.detector.PanelList()

    # Check all defaults
    assert(pl.name == "")
    assert(pl.n_panels == 0)
    assert(pl.geometry_hash is None)

    p1 = panel1_setup()
    p2 = panel2_setup()

    pl.append(p1)
    pl.append(p2)

    # Check that links to the PanelList are created
    assert(p1.panellist is not None)
    assert(p2.panellist is not None)
    assert(p1.panellist is p2.panellist)

    # Misc checks
    assert(pl.n_panels == 2)

    ###########################################################################
    # Test the manipulation of data
    ###########################################################################

    d0 = pl[0].zeros()
    d1 = pl[1].ones()
    dl = pl.concatentate_panel_data([d0, d1])
    assert(all(dl[0:(pl[0].n_pixels - 1)] == 0))
    assert(all(dl[(pl[0].n_pixels):(pl.n_pixels - 1)] == 1))
    del d0, d1

    d0a = pl.get_panel_data(0, dl)
    d1a = pl.get_panel_data(1, dl)
    assert(all(d0a == 0))
    assert(all(d1a == 1))
    del d0a, d1a

    ds = pl.split_panel_data(dl)
    assert(all(ds[0] == 0))
    assert(all(ds[1] == 1))
    del ds

    dp = pl.zeros()
    pl.put_panel_data(0, pl[0].ones(), dp)
    pl.put_panel_data(1, pl[1].zeros(), dp)
    assert(all(dp[0:(pl[0].n_pixels - 1)] == 1))
    assert(all(dp[(pl[0].n_pixels):(pl.n_pixels - 1)] == 0))
    del dp, pl

    ###########################################################################
    # Test cache creation/deletion
    ###########################################################################

    def panellist_setup():
        p0 = panel1_setup()
        p1 = panel2_setup()
        pl = ba.detector.PanelList()
        pl.append(p0)
        pl.append(p1)
        pl.beam.wavelength = 0.1
        return pl
    # Check that cache is cleared

    def cache_cleared(pl):
        assert(pl._v is None)
        assert(pl._sa is None)
        assert(pl._pf is None)
        assert(pl._k is None)
        assert(pl._ps is None)
        assert(pl._rsbb is None)
        assert(pl._gh is None)
    # Check that cache is created

    def cache_created(pl):
        assert(pl._v is not None)
        assert(pl._sa is not None)
        assert(pl._pf is not None)
        assert(pl._k is not None)
        assert(pl._ps is not None)
        assert(pl._rsbb is not None)
        assert(pl._gh is not None)
    # Load cache by requesting arrays

    def create_cache(pl):
        V = pl.V
        sa = pl.solid_angle
        pf = pl.polarization_factor
        q = pl.Q
        ps = pl.pixel_size
        rsbb = pl.real_space_bounding_box
        gh = pl.geometry_hash

    pl = panellist_setup()
    cache_cleared(pl)
    create_cache(pl)
    cache_created(pl)

    pl = panellist_setup()
    create_cache(pl)
    pl[0].F = (1, 0, 0)  # This should clear the cache
    cache_cleared(pl)
    pl = panellist_setup()
    create_cache(pl)
    pl[0].S = (1, 0, 0)  # This should clear the cache
    cache_cleared(pl)
    pl = panellist_setup()
    create_cache(pl)
    pl[0].T = (1, 0, 0)  # This should clear the cache
    cache_cleared(pl)
    pl = panellist_setup()
    create_cache(pl)
    pl[0].nF = 5  # This should clear the cache
    cache_cleared(pl)
    pl = panellist_setup()
    create_cache(pl)
    pl[0].nS = 5  # This should clear the cache
    cache_cleared(pl)
    del pl

    ###########################################################################
    # Test that geometry hash works
    ###########################################################################

    # Geometry hash is None by default
    pl = ba.detector.PanelList()
    assert(pl.geometry_hash is None)

    # Self consistency
    pl = panellist_setup()
    assert(pl.geometry_hash == pl.geometry_hash)

    # Comparison between two identical setups
    pl1 = panellist_setup()
    pl2 = panellist_setup()
    assert(pl1.geometry_hash == pl2.geometry_hash)

    # Check mismatch if something is changed
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[0].nF += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[0].nS += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[0].F += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[0].S += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[0].T += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[1].nF += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[1].nS += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[1].F += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[1].S += 1
    assert(p1.geometry_hash != p2.geometry_hash)
    p1 = panellist_setup()
    p2 = panellist_setup()
    p2[1].T += 1
    assert(p1.geometry_hash != p2.geometry_hash)


# def test_set_data(main=False):
#
#     pl = ba.detector.PanelList()
#     p = ba.detector.Panel()
#     p.simple_setup(100, 101, 100e-6, 1, 1.5e-10)
#     p.data = np.ones([p.nS, p.nF])
#     pl.append(p)
#     p = ba.detector.Panel()
#     p.simple_setup(102, 103, 100e-6, 1, 1.5e-10)
#     p.data = np.zeros([p.nS, p.nF])
#     pl.append(p)
#
#     a = pl.data
#     pl.data = a
#
#     q = pl.Q
#     q = pl[0].Q
#
#     assert(np.max(np.abs(a - pl.data)) == 0)


# def test_beam(main=False):
#
#     p = ba.detector.Panel()
#     wavelength = 1
#     p.beam.wavelength = wavelength
#
#     assert(p.beam.wavelength == wavelength)
#
#     pl = ba.detector.PanelList()
#     pl.wavelength = 1
#     assert(pl.beam.wavelength == wavelength)


# def test_reshape(main=False):
#
#     p = ba.detector.Panel()
#     nF = 3
#     nS = 4
#     p.simple_setup(nF=nF, nS=nS, pixel_size=1, distance=1, wavelength=1)
#     d = np.arange(0, nF * nS)
#     p.data = d.copy()
#     d = p.reshape(d)
#     assert(d.shape[1] == nF)
#     assert(d.shape[0] == nS)


# def test_panellist_simple_setup(main=False):
#
#     pl = ba.detector.PanelList()
#     pl.simple_setup(nF=nF,nS=nS,pixel_size=1,distance=1,wavelength=1)
#     assert(pl[0].V[0,0] == -nF/2.0 + 0.5)
#     assert(pl[0].V[-1,0] == nF/2.0 - 0.5)
#     assert(pl[0].V[0,1] == -nS/2.0 + 0.5)
#     assert(pl[0].V[-1,1] == nS/2.0 - 0.5)
#     assert(pl.V[0,0] == -nF/2.0 + 0.5)
#     assert(pl.V[-1,0] == nF/2.0 - 0.5)
#     assert(pl.V[0,1] == -nS/2.0 + 0.5)
#     assert(pl.V[-1,1] == nS/2.0 - 0.5)


if __name__ == "__main__":

    main = True
#     test_set_data(main)
    test_panel(main)
    test_panellist(main)
#     test_beam(main)
#     test_reshape(main)

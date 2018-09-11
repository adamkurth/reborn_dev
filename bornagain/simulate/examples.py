import pkg_resources
import numpy as np
from bornagain import detector
from bornagain.units import hc
from bornagain.external import crystfel
from bornagain.simulate import atoms
from bornagain.target.crystal import structure
from bornagain.simulate.clcore import ClCore


lysozyme_pdb_file = pkg_resources.resource_filename('bornagain.simulate', 'data/pdb/2LYZ.pdb')
pnccd_geom_file = pkg_resources.resource_filename('bornagain.simulate', 'data/geom/pnccd_front.geom')
cspad_geom_file = pkg_resources.resource_filename('bornagain.simulate', 'data/geom/cspad.geom')


def pnccd_pads():

    r"""

    Generate a list of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances that are inspired by
    the `pnCCD <https://doi.org/10.1016/j.nima.2009.12.053>`_ detector.

    Returns: List of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances.

    """

    return crystfel.geometry_file_to_pad_geometry_list(pnccd_geom_file)


def cspad_pads():

    r"""

    Generate a list of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances that are inspired by
    the `CSPAD <http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-15284.pdf>`_ detector.

    Returns: List of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances.

    """

    return crystfel.geometry_file_to_pad_geometry_list(cspad_geom_file)


def lysozyme_molecule(pads=None, wavelength=None):

    r"""

    Simple simulation of lysozyme molecule using :class:`ClCore <bornagain.simulate.clcore.ClCore>`.

    Args:
        pads: List of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances.
        wavelength: As always, in SI units.

    Returns: dictionary with {'pad_geometry': pads, 'intensity': data_list}

    """

    if wavelength is None:
        wavelength = 1.5e-10

    photon_energy = hc / wavelength

    if pads is None:
        pads = crystfel.geometry_file_to_pad_geometry_list(pnccd_geom_file)

    sim = ClCore(group_size=32, double_precision=False)

    cryst = structure(lysozyme_pdb_file)
    r = cryst.r
    f = atoms.get_scattering_factors(cryst.Z, photon_energy=photon_energy)
    q = [pad.q_vecs(beam_vec=[0, 0, 1], wavelength=wavelength) for pad in pads]
    q = np.ravel(q)

    A = sim.phase_factor_qrf(q, r, f)
    I = np.abs(A)**2

    data_list = detector.split_pad_data(pads, I)

    return {'pad_geometry': pads, 'intensity': data_list}
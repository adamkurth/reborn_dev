"""

Some simple examples for testing purposes.

Don't build any of this into your code.

"""

import pkg_resources
import numpy as np
import bornagain as ba
from bornagain import detector
from bornagain.units import hc
from bornagain import utils
from bornagain.external import crystfel
from bornagain.simulate import atoms
from bornagain.target.crystal import Structure
from bornagain.simulate.clcore import ClCore


lysozyme_pdb_file = pkg_resources.resource_filename('bornagain.simulate', 'data/pdb/2LYZ.pdb')
psi_pdb_file = pkg_resources.resource_filename('bornagain.simulate', 'data/pdb/1jb0.pdb')
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


def lysozyme_molecule(pads=None, wavelength=None, random_rotation=False):

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

    cryst = Structure(lysozyme_pdb_file)
    r = cryst.r
    f = atoms.get_scattering_factors(cryst.Z, photon_energy=photon_energy)
    q = [pad.q_vecs(beam_vec=[0, 0, 1], wavelength=wavelength) for pad in pads]
    q = np.ravel(q)


    if random_rotation:
        R = utils.random_rotation()
    else:
        R = None

    A = sim.phase_factor_qrf(q, r, f, R)
    I = np.abs(A)**2

    data_list = detector.split_pad_data(pads, I)

    return {'pad_geometry': pads, 'intensity': data_list}


class PDBMoleculeSimulator(object):

    r"""

    A simple generator of simulated single-molecule intensities, from a pdb file.  Defaults to lysozyme at 1.5 A
    wavelength, on a pnccd detector layout.

    """

    def __init__(self, pdb_file=None, pads=None, wavelength=None, random_rotation=False):

        r"""

        This will setup the opencl simulation core.

        Args:
            pdb_file: path to a pdb file
            pads: array of :class:`PADGeometry bornagain.detector.PADGeometry` intances
            wavelength: in SI units of course
            random_rotation: True or False
        """

        if pdb_file is None:
            pdb_file = lysozyme_pdb_file

        if pads is None:
            pads = crystfel.geometry_file_to_pad_geometry_list(pnccd_geom_file)

        if wavelength is None:
            wavelength = 1.5e-10

        photon_energy = hc / wavelength

        self.clcore = ClCore(group_size=32, double_precision=False)
        cryst = Structure(pdb_file)

        self.random_rotation = random_rotation

        r = cryst.r
        f = atoms.get_scattering_factors(cryst.Z, photon_energy=photon_energy)
        q = [pad.q_vecs(beam_vec=[0, 0, 1], wavelength=wavelength) for pad in pads]
        q = np.ravel(q)

        self.q_gpu = self.clcore.to_device(q)
        self.r_gpu = self.clcore.to_device(r)
        self.f_gpu = self.clcore.to_device(f)
        self.a_gpu = self.clcore.to_device(shape=(self.q_gpu.shape[0]), dtype=self.clcore.complex_t)

    def next(self):

        r"""

        Generate another simulated pattern.  No scaling or noise added.

        Returns: Flat array of diffraction intensities.

        """

        if self.random_rotation:
            R = utils.random_rotation()
        else:
            R = None

        self.clcore.phase_factor_qrf(self.q_gpu, self.r_gpu, self.f_gpu, R, self.a_gpu)
        I = self.a_gpu.get()
        if ba.get_global('debug') > 0:
            print(I, I.shape, type(I), I.dtype, np.max(I))
            print(self.q_gpu.shape)
        return np.abs(I)**2
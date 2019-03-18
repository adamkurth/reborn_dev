"""

Some simple examples for testing purposes.

Don't build any of this into your code.

"""

import time
import pkg_resources
import numpy as np
import bornagain as ba
from bornagain import detector
from bornagain.units import hc, r_e
from bornagain import utils
from bornagain.external import crystfel
from bornagain.simulate import atoms
from bornagain.target.crystal import CrystalStructure
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


def lysozyme_molecule(pad_geometry=None, wavelength=1.5e-10, random_rotation=False):

    r"""

    Simple simulation of lysozyme molecule using :class:`ClCore <bornagain.simulate.clcore.ClCore>`.

    Args:
        pad_geometry: List of :class:`PADGeometry <bornagain.detector.PADGeometry>` instances.
        wavelength: As always, in SI units.

    Returns: dictionary with {'pad_geometry': pads, 'intensity': data_list}

    """

    photon_energy = hc / wavelength

    if pad_geometry is None:
        pad_geometry = crystfel.geometry_file_to_pad_geometry_list(cspad_geom_file)

    sim = ClCore(group_size=32, double_precision=False)

    cryst = CrystalStructure(lysozyme_pdb_file)
    r = cryst.r
    f = atoms.get_scattering_factors(cryst.Z, photon_energy=photon_energy)
    q = [pad.q_vecs(beam_vec=[0, 0, 1], wavelength=wavelength) for pad in pad_geometry]
    q = np.ravel(q)


    if random_rotation:
        R = utils.random_rotation()
    else:
        R = None

    A = sim.phase_factor_qrf(q, r, f, R)
    I = np.abs(A)**2

    data_list = detector.split_pad_data(pad_geometry, I)

    return {'pad_geometry': pad_geometry, 'intensity': data_list}


class PDBMoleculeSimulator(object):

    r"""

    A simple generator of simulated single-molecule intensities, from a pdb file.  Defaults to lysozyme at 1.5 A
    wavelength, on a pnccd detector layout.

    """

    def __init__(self, pdb_file=None, pad_geometry=None, wavelength=1.5e-10, random_rotation=True):

        r"""

        This will setup the opencl simulation core.

        Args:
            pdb_file: path to a pdb file
            pad_geometry: array of :class:`PADGeometry bornagain.detector.PADGeometry` intances
            wavelength: in SI units of course
            random_rotation: True or False
        """

        if pdb_file is None:
            pdb_file = lysozyme_pdb_file

        if pad_geometry is None:
            pad_geometry = crystfel.geometry_file_to_pad_geometry_list(cspad_geom_file)

        photon_energy = hc / wavelength

        self.clcore = ClCore(group_size=32)
        cryst = CrystalStructure(pdb_file)

        self.random_rotation = random_rotation

        r = cryst.r
        f = atoms.get_scattering_factors(cryst.Z, photon_energy=photon_energy)
        q = [pad.q_vecs(beam_vec=[0, 0, 1], wavelength=wavelength) for pad in pad_geometry]
        q = np.ravel(q)
        nq = int(len(q)/3)

        self.q_gpu = self.clcore.to_device(q)
        self.r_gpu = self.clcore.to_device(r)
        self.f_gpu = self.clcore.to_device(f)
        self.a_gpu = self.clcore.to_device(shape=(nq,), dtype=self.clcore.complex_t)

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


class CrystalSimulatorV1(object):

    def __init__(self, pad_geometry=None, beam=None, crystal_structure=None, n_iterations=1, random_rotation=True,
                 approximate_shape_transform=True, cromer_mann=False, expand_symmetry=False,
                 cl_double_precision=False, cl_group_size=32, poisson_noise=False):

        self.pad_geometry = pad_geometry
        self.beam = beam
        self.crystal_structure = crystal_structure
        self.n_iterations = n_iterations
        self.random_rotation = random_rotation
        self.cromer_mann = cromer_mann
        self.poisson_noise = poisson_noise

        self.q = self.pad_geometry.q_vecs(beam=beam)
        self.qmag = self.pad_geometry.q_mags(beam=beam)
        self.sa = self.pad_geometry.solid_angles()
        self.pol = self.pad_geometry.polarization_factors(beam=beam)

        if expand_symmetry:
            self.r = self.crystal_structure.get_symmetry_expanded_coordinates()
            Z = self.crystal_structure.molecule.atomic_numbers
            self.Z = np.concatenate([Z] * self.crystal_structure.spacegroup.n_molecules)
        else:
            self.r = self.crystal_structure.molecule.coordinates
            self.Z = self.crystal_structure.molecule.atomic_numbers

        self.f = ba.simulate.atoms.get_scattering_factors(self.Z, photon_energy=beam.photon_energy)

        self.clcore = ClCore(group_size=cl_group_size, double_precision=cl_double_precision)
        self.r_dev = self.clcore.to_device(self.r, dtype=self.clcore.real_t)
        self.f_dev = self.clcore.to_device(self.f, dtype=self.clcore.complex_t)
        self.F_dev = self.clcore.to_device(shape=self.pad_geometry.shape(), dtype=self.clcore.complex_t)
        self.S2_dev = self.clcore.to_device(shape=self.pad_geometry.shape(), dtype=self.clcore.real_t)

        if approximate_shape_transform:
            self.shape_transform = self.clcore.gaussian_lattice_transform_intensities_pad
        else:
            self.shape_transform = self.clcore.lattice_transform_intensities_pad

        self.beam_area = np.pi * self.beam.diameter_fwhm ** 2 / 4.0
        self.cell_volume = self.crystal_structure.unitcell.volume
        self.solid_angles = self.pad_geometry.solid_angles()
        self.polarization_factor = self.pad_geometry.polarization_factors(beam=self.beam)
        self.intensity_prefactor = self.beam.photon_number_fluence * r_e ** 2 * self.solid_angles * self.polarization_factor
        self.intensity_prefactor = self.pad_geometry.reshape(self.intensity_prefactor)

    def generate_pattern(self, rotation_matrix=None):

        cryst = self.crystal_structure
        beam = self.beam
        pad = self.pad_geometry

        this_mosaic_domain_size = np.random.normal(cryst.mosaic_domain_size, cryst.mosaic_domain_size_fwhm / 2.354820045)
        this_crystal_size = np.random.normal(cryst.crystal_size, cryst.crystal_size_fwhm / 2.354820045)

        n_cells_whole_crystal = \
            np.ceil(min(self.beam_area, this_crystal_size ** 2) * this_crystal_size / self.cell_volume)
        n_cells_mosaic_domain = \
            np.ceil(min(self.beam_area, this_mosaic_domain_size ** 2) * this_mosaic_domain_size / self.cell_volume)

        if self.random_rotation:
            R = ba.utils.random_rotation()
        else:
            R = rotation_matrix

        if not self.cromer_mann:
            self.clcore.phase_factor_pad(self.r_dev, self.f_dev, self.pad_geometry.t_vec, self.pad_geometry.fs_vec,
                                         self.pad_geometry.ss_vec, beam.beam_vec, self.pad_geometry.n_fs,
                                         self.pad_geometry.n_ss, beam.wavelength, R, self.F_dev, add=False)
            F2 = np.abs(self.F_dev.get()) ** 2
        else:
            raise ValueError('Cromer-Mann needs to be re-implemented')

        self.S2_dev *= 0

        for _ in np.arange(1, (self.n_iterations + 1)):

            B = ba.utils.random_beam_vector(beam.beam_divergence_fwhm)
            w = hc / np.random.normal(beam.photon_energy, beam.photon_energy_fwhm / 2.354820045)
            Rm = ba.utils.random_mosaic_rotation(cryst.mosaicity_fwhm).dot(R)
            T = pad.t_vec + pad.fs_vec * (np.random.random([1]) - 0.5) + pad.ss_vec * (np.random.random([1]) - 0.5)

            self.shape_transform(cryst.unitcell.o_mat.T.copy(), np.array([np.ceil(n_cells_mosaic_domain ** (1 / 3.))] * 3),
                                 T, pad.fs_vec, pad.ss_vec, B, pad.n_fs, pad.n_ss, w, Rm, self.S2_dev, add=True)

        S2 = self.S2_dev.get() / self.n_iterations
        intensity = self.intensity_prefactor * F2 * S2 * n_cells_whole_crystal / n_cells_mosaic_domain

        if self.poisson_noise:
            intensity = np.random.poisson(intensity)

        return intensity

"""

Some simple examples for testing purposes.

Don't build any of this into your code.

"""

import pkg_resources
import numpy as np
from scipy.spatial.transform import Rotation
from .. import detector
from ..utils import rotation_about_axis, random_unit_vector, random_beam_vector, max_pair_distance, ensure_list
from . import atoms
from . import solutions
from ..target.crystal import CrystalStructure
from .clcore import ClCore
from scipy import constants as const

hc = const.h*const.c
r_e = const.value('classical electron radius')

lysozyme_pdb_file = pkg_resources.resource_filename('reborn', 'data/pdb/2LYZ.pdb')
psi_pdb_file = pkg_resources.resource_filename('reborn', 'data/pdb/1jb0.pdb')
pnccd_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/pnccd_front_geometry.json')
cspad_geom_file = pkg_resources.resource_filename('reborn', 'data/geom/cspad_geometry.json')


def pnccd_pads():
    r"""
    Generate a list of :class:`PADGeometry <reborn.detector.PADGeometry>` instances that are inspired by
    the `pnCCD <https://doi.org/10.1016/j.nima.2009.12.053>`_ detector.

    Returns: List of |PADGeometry| instances
    """
    pads = detector.load_pad_geometry_list(pnccd_geom_file)
    for p in pads:
        p.t_vec[2] = 0.1
    return pads


def cspad_pads(detector_distance=0.1):
    r"""
    Generate a list of |PADGeometry| instances that are inspired by the |CSPAD| detector.

    Arguments:
        detector_distance (float): Detector distance in SI units

    Returns: List of |PADGeometry| instances
    """
    pads = detector.load_pad_geometry_list(cspad_geom_file)
    for p in pads:
        p.t_vec[2] = detector_distance
    return pads


def jungfrau4m_pads(detector_distance=0.1, binning=1):
    r"""
    Generate a list of |PADGeometry| instances that are inspired by the |Jungfrau| 4M detector.

    Arguments:
        detector_distance (float): Detector distance in SI units
        binning (int): Bin the detector into larger NxN virtual pixels.

    Returns: List of |PADGeometry| instances
    """
    pads = detector.tiled_pad_geometry_list(pad_shape=(int(512/binning), int(1024/binning)), pixel_size=75e-6*binning,
                                            distance=detector_distance, tiling_shape=(4, 2), pad_gap=36 * 75e-6)
    gap = 9e-3
    pads[0].t_vec += + np.array([1, 0, 0]) * gap / 2 - np.array([0, 1, 0]) * gap / 2
    pads[1].t_vec += + np.array([1, 0, 0]) * gap / 2 - np.array([0, 1, 0]) * gap / 2
    pads[2].t_vec += - np.array([1, 0, 0]) * gap / 2 - np.array([0, 1, 0]) * gap / 2
    pads[3].t_vec += - np.array([1, 0, 0]) * gap / 2 - np.array([0, 1, 0]) * gap / 2
    pads[4].t_vec += + np.array([1, 0, 0]) * gap / 2 + np.array([0, 1, 0]) * gap / 2
    pads[5].t_vec += + np.array([1, 0, 0]) * gap / 2 + np.array([0, 1, 0]) * gap / 2
    pads[6].t_vec += - np.array([1, 0, 0]) * gap / 2 + np.array([0, 1, 0]) * gap / 2
    pads[7].t_vec += - np.array([1, 0, 0]) * gap / 2 + np.array([0, 1, 0]) * gap / 2
    return pads

def simulate_water(pad_geometry=None, beam=None, water_thickness=1e-6):
    r"""
    Simulate water scatter.  Takes a PAD geometry and beam specification and returns list of 2D numpy arrays with the
    scattering intensity in photon units.

    Args:
        pad_geometry (list of |PADGeometry|'s): List of PAD geometry specifications.
        beam (|Beam|): Beam specification
        water_thickness (float): Thickness of water in SI units.

    Returns:
        list of 2D numpy arrays
    """
    n_water_molecules = water_thickness * np.pi * (beam.diameter_fwhm / 2) ** 2 * solutions.water_number_density()
    pads = ensure_list(pad_geometry)
    q_mags = [p.q_mags(beam=beam) for p in pads]
    J = beam.photon_number_fluence
    P = detector.concat_pad_data([p.polarization_factors(beam=beam) for p in pads])
    SA = detector.concat_pad_data([p.solid_angles() for p in pads])
    F_water = solutions.get_water_profile(q_mags)
    F2_water = F_water ** 2 * n_water_molecules
    I = r_e ** 2 * J * P * SA * F2_water
    return detector.split_pad_data(pads, I)

def lysozyme_molecule(pad_geometry=None, wavelength=1.5e-10, random_rotation=True):

    r"""

    Simple simulation of lysozyme molecule using :class:`ClCore <reborn.simulate.clcore.ClCore>`.

    Arguments:
        pad_geometry: List of :class:`PADGeometry <reborn.detector.PADGeometry>` instances.
        wavelength: As always, in SI units.
        random_rotation: If True generate a random rotation.  Default is True.

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
        rot = Rotation.random().as_matrix()
    else:
        rot = None

    amps = sim.phase_factor_qrf(q, r, f, rot)
    intensity = np.abs(amps)**2

    data_list = detector.split_pad_data(pad_geometry, intensity)

    return {'pad_geometry': pad_geometry, 'intensity': data_list}


class PDBMoleculeSimulator(object):

    r"""

    A simple generator of simulated single-molecule intensities, from a pdb file.  Defaults to lysozyme at 1.5 A
    wavelength, on a pnccd detector layout.

    """

    def __init__(self, pdb_file=None, pad_geometry=None, wavelength=1.5e-10, random_rotation=True):

        r"""

        This will setup the opencl simulation core.

        Arguments:
            pdb_file: path to a pdb file
            pad_geometry: array of :class:`PADGeometry <reborn.detector.PADGeometry>` intances
            wavelength: in SI units of course
            random_rotation (bool): True or False
        """

        if pdb_file is None:
            pdb_file = lysozyme_pdb_file

        if pad_geometry is None:
            pad_geometry = crystfel.geometry_file_to_pad_geometry_list(cspad_geom_file)

        photon_energy = hc / wavelength

        self.clcore = ClCore(group_size=32)
        cryst = CrystalStructure(pdb_file)

        r = cryst.molecule.coordinates
        f = cryst.molecule.get_scattering_factors(photon_energy=photon_energy)
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
            rot = Rotation.random().as_matrix()
        else:
            rot = None

        self.clcore.phase_factor_qrf(self.q_gpu, self.r_gpu, self.f_gpu, rot, self.a_gpu)
        intensity = self.a_gpu.get()
        return np.abs(intensity)**2


class MoleculeSimulatorV1(object):

    def __init__(self, beam=None, molecule=None, pad=None, oversample=10, max_mesh_size=200, clcore=None):

        if clcore is None:
            self.clcore = ClCore(group_size=32)
        else:
            self.clcore = clcore
        self.pad = pad
        self.molecule = molecule
        self.beam = beam
        self.oversample = oversample
        self.q_vecs = pad.q_vecs(beam=beam)
        self.f = molecule.get_scattering_factors(beam=beam)
        self.intensity_prefactor = pad.reshape(beam.photon_number_fluence * r_e ** 2 * pad.solid_angles() *
                                          pad.polarization_factors(beam=beam))
        self.resolution = pad.max_resolution(beam=beam)
        self.mol_size = max_pair_distance(molecule.coordinates)
        self.qmax = 2 * np.pi / self.resolution
        self.mesh_size = int(np.ceil(10 * self.mol_size / self.resolution))
        self.mesh_size = int(min(self.mesh_size, max_mesh_size))
        self.a_map_dev = clcore.to_device(shape=(self.mesh_size ** 3,), dtype=clcore.complex_t)
        self.q_dev = clcore.to_device(self.q_vecs, dtype=clcore.real_t)
        self.a_out_dev = clcore.to_device(dtype=clcore.complex_t, shape=pad.shape())
        clcore.phase_factor_mesh(self.molecule.coordinates, self.f, N=self.mesh_size, q_min=-self.qmax,
                                 q_max=self.qmax, a=self.a_map_dev)

    def generate_pattern(self, rotation=None, poisson=False):

        if rotation is None:
            rotation = Rotation.random().as_matrix()
        self.clcore.mesh_interpolation(self.a_map_dev, self.q_dev, N=self.mesh_size, q_min=-self.qmax,
                                       q_max=self.qmax, R=rotation, a=self.a_out_dev)
        if poisson:
            intensity = np.random.poisson(self.intensity_prefactor * np.abs(self.a_out_dev.get()) ** 2)
        else:
            intensity = self.intensity_prefactor * np.abs(self.a_out_dev.get()) ** 2
        return intensity


class CrystalSimulatorV1(object):

    r"""

    Class for generating crystal diffraction patterns.  Generates the average pattern upon:

    1) Randomizing the x-ray beam direction (beam divergence)
    2) Randomizing the outgoing beam according to pixel area ("pixel solid angle")
    3) Randomizing photon energy (spectral width)
    4) Randomizing the orientation of crystal mosaic domains (mosaicity)
    5) Randomizing the shape transforms or Gaussian crystal-size broadening

    Computations are done on a GPU with OpenCL.

    """

    def __init__(self, pad_geometry=None, beam=None, crystal_structure=None, n_iterations=1,
                 approximate_shape_transform=True, expand_symmetry=False,
                 cl_double_precision=False, cl_group_size=32, poisson_noise=False):

        r"""

        Arguments:
            pad_geometry (list of :class:`PADGeometry <reborn.detector.PADGeometry>` instances): PAD geometry.
            beam (:class:`Beam <reborn.source.Beam>`): A beam instance.
            crystal_structure (:class:`CrystalStructure <reborn.target.crystal.CrystalStructure>`): A crystal
                              structure.
            n_iterations (int): Number of iterations to average over
            approximate_shape_transform (bool): Use a Gaussian approximation to shape transforms, else use the analytic
                                         parallelepiped shape transform formula.
            expand_symmetry (bool): Duplicate the asymmetric unit according to spacegroup symmetry in crystal_structure.
            cl_double_precision (bool): Use double precision if available on GPU device.
            cl_group_size (int): GPU group size (see the :class:`ClCore <reborn.simulate.clcore.ClCore>` class).
            poisson_noise (bool): Add Poisson noise to the resulting pattern.

        """

        if not isinstance(pad_geometry, list):
            pad_geometry = [pad_geometry]
        self.pad_geometry = pad_geometry
        self.beam = beam
        self.crystal_structure = crystal_structure
        self.n_iterations = n_iterations
        self.poisson_noise = poisson_noise

        self.q = []
        self.qmag = []
        # self.sa = []
        # self.pol = []
        self.ipf = []
        for p in pad_geometry:
            self.q.append(p.q_vecs(beam=beam))
            self.qmag.append(p.q_mags(beam=beam))
            # self.sa.append(p.solid_angles())
            # self.pol.append(p.polarization_factors(beam=beam))
            ipf = self.beam.photon_number_fluence*r_e**2*p.solid_angles()*p.polarization_factors(beam=beam)
            self.ipf.append(p.reshape(ipf))

        if expand_symmetry:
            self.r = self.crystal_structure.get_symmetry_expanded_coordinates()
            atom_z = self.crystal_structure.molecule.atomic_numbers
            self.Z = np.concatenate([atom_z] * self.crystal_structure.spacegroup.n_molecules)
        else:
            self.r = self.crystal_structure.molecule.coordinates
            self.Z = self.crystal_structure.molecule.atomic_numbers
        self.f = atoms.get_scattering_factors(self.Z, photon_energy=beam.photon_energy)

        self.clcore = ClCore(group_size=cl_group_size, double_precision=cl_double_precision)
        self.r_dev = self.clcore.to_device(self.r, dtype=self.clcore.real_t)
        self.f_dev = self.clcore.to_device(self.f, dtype=self.clcore.complex_t)
        self.F_dev = []
        self.S2_dev = []
        for p in pad_geometry:
            self.F_dev.append(self.clcore.to_device(shape=p.shape(), dtype=self.clcore.complex_t))
            self.S2_dev.append(self.clcore.to_device(shape=p.shape(), dtype=self.clcore.real_t))

        if approximate_shape_transform:
            self.shape_transform = self.clcore.gaussian_lattice_transform_intensities_pad
        else:
            self.shape_transform = self.clcore.lattice_transform_intensities_pad

        self.beam_area = np.pi * self.beam.diameter_fwhm ** 2 / 4.0
        self.cell_volume = self.crystal_structure.unitcell.volume

    def generate_pattern(self, rotation_matrix=None):

        r"""
        Arguments:
            rotation_matrix: Specify a rotation matrix, else a random rotation is generated.

        Returns: A numpy array with diffraction intensities
        """

        cryst = self.crystal_structure
        beam = self.beam
        pad = self.pad_geometry

        this_mosaic_domain_size = np.random.normal(cryst.mosaic_domain_size, cryst.mosaic_domain_size_fwhm / 2.354820045)
        this_crystal_size = np.random.normal(cryst.crystal_size, cryst.crystal_size_fwhm / 2.354820045)

        n_cells_whole_crystal = \
            np.ceil(min(self.beam_area, this_crystal_size ** 2) * this_crystal_size / self.cell_volume)
        n_cells_mosaic_domain = \
            np.ceil(min(self.beam_area, this_mosaic_domain_size ** 2) * this_mosaic_domain_size / self.cell_volume)

        if rotation_matrix is None:
            rotation_matrix = Rotation.random().as_matrix()

        moltrans = []
        for i in range(len(self.pad_geometry)):
            self.clcore.phase_factor_pad(self.r_dev, f=self.f_dev, beam=beam, pad=self.pad_geometry[i],
                                         R=rotation_matrix, a=self.F_dev[i], add=False)
            moltrans.append(np.abs(self.F_dev[i].get()) ** 2)
            self.S2_dev[i] *= 0

        for _ in np.arange(1, (self.n_iterations + 1)):

            # Random incoming beam vector
            b_in = random_beam_vector(beam.beam_divergence_fwhm)
            # Random wavelength
            wav = hc / np.random.normal(beam.photon_energy, beam.photon_energy_fwhm / 2.354820045)
            # Random crystal mosaic domain rotation
            rot = np.dot(rotation_about_axis(cryst.mosaicity_fwhm/2.354820045*np.random.normal(), random_unit_vector()),
                         rotation_matrix)
            # Random location within pixels (pixel solid angle)
            osfs = np.random.random([1]) - 0.5
            osss = np.random.random([1]) - 0.5

            for i in range(len(self.pad_geometry)):
                pad = self.pad_geometry[i]
                t_vec = pad.t_vec + pad.fs_vec * osfs + pad.ss_vec * osss

                self.shape_transform(cryst.unitcell.o_mat.T, np.array([np.ceil(n_cells_mosaic_domain ** (1 / 3.))] * 3),
                                     t_vec, pad.fs_vec, pad.ss_vec, b_in, pad.n_fs, pad.n_ss, wav, rot, self.S2_dev[i],
                                     add=True)

        intensity = []
        for i in range(len(self.pad_geometry)):
            shapetrans = self.S2_dev[i].get() / self.n_iterations
            intensity.append(moltrans[i] * shapetrans * self.ipf[i] * n_cells_whole_crystal / n_cells_mosaic_domain)

            if self.poisson_noise:
                intensity[i] = np.random.poisson(intensity[i])

        return intensity

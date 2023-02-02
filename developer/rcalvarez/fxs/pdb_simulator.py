import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from glob import glob
from reborn.dataframe import DataFrame
from reborn.detector import jungfrau4m_pad_geometry_list
from reborn.fileio.getters import FrameGetter
from reborn.simulate import clcore
from reborn.simulate.gas import get_gas_background
from reborn.simulate.solutions import water_scattering_factor_squared
from reborn.source import Beam
from reborn.target import atoms
from reborn.target import crystal


class Simulator(FrameGetter):
    r"""
    Molecule diffraction from a PDB file simulation FrameGetter.

    Arguments
    ---------
        pad_geometry (|PADGeometryList|): Detector geometry
        beam (|Beam|): X-ray beam
        kwargs:
            experiment_id (str): Simulation identifier (default is 'simulation')
            run_id (int): Simulation run identifier (default is 0)
            n_frames (int): Number of simulations to run (default is 1000)
            pdb (str): PDB id to simulate (this will setup crystal object for you)
            crystal (|CrystalStructure|): Crystal object (only use this if you need to do
                                          something with the crystal, i.e. enable molecular machine etc.)
            n_particles (int): Number of particles in the simulation. (default is 1)
            random_seed (int): Seed if reproducible random number are needed (default is None)
            sheets (bool): Add solution background from sheet jet (default is False)
                sheet_thickness (float): Thickness of sheet jet (default is 1e-6)
                water_temperature (float): Temperature of sheet jet (default is 298.0 K)
            droplets (bool): Add solution background from droplet (default is False)
                droplet_diameter (bool): Diameter of water droplet (default is 0.5e-6)
                water_temperature (bool): Temperature of droplet (default is 298 K)
            gas_background (bool): Add gas background (default is False)
                gas_type (str): Type of gas background ('helium', 'he', or 'air'; default is 'he')
                gas_path (int): Length of full gas path (default is 1 m)
                gas_pressure (float): Simulated gas pressure (default 101325.0 Pa)
                gas_temperature (float): Simulated gas temperature (default is 293.15 K)
            poisson_noise (bool): Add Poisson noise to simulation (default is False)
        """
    def __init__(self, pad_geometry, beam, **kwargs):
        super().__init__()
        self.experiment_id = kwargs.get('experiment_id', 'simulation')
        self.run_id = kwargs.get('run_id', 0)
        self.n_frames = kwargs.get('n_frames', 1000)
        self.n_particles = kwargs.get('n_particles', 1)  # 1,2,10,100

        random_seed = kwargs.get('random_seed', None)

        if random_seed is not None:
            np.random.seed(random_seed)

        self.beam = beam
        self.fluence = self.beam.photon_number_fluence
        self.photon_energy = self.beam.photon_energy

        self.pads = pad_geometry
        self.q_vecs = self.pads.q_vecs(beam=self.beam)
        self.q_mags = self.pads.q_mags(beam=self.beam)
        self.f2phot = self.pads.f2phot(beam=self.beam)

        pdb = kwargs.get('pdb', None)
        cryst = kwargs.get('crystal', None)
        if cryst is None:
            if pdb is None:
                raise ValueError('Please provide pdb string or CrystalStructure.')
            cryst = crystal.CrystalStructure(pdb)
        r_vecs = cryst.molecule.coordinates
        r_vecs -= np.mean(r_vecs, axis=0)
        atomic_numbers = cryst.molecule.atomic_numbers

        self.simulation_engine = clcore.ClCore()
        uniq_z = np.unique(atomic_numbers)
        grv = []
        gfs = []
        for z in uniq_z:
            grv.append(np.squeeze(r_vecs[np.where(atomic_numbers == z), :]))
            gfs.append(atoms.hubbel_henke_scattering_factors(q_mags=self.q_mags,
                                                             photon_energy=self.photon_energy,
                                                             atomic_number=z))
        self.grouped_r_vecs = grv
        self.grouped_fs = gfs

        self.sheets = kwargs.get('sheets', False)
        if self.sheets:
            sheet_thickness = kwargs.get('sheet_thickness', 1e-6)
            self.water_temperature = kwargs.get('water_temperature', 298.0)
            self.sheet_volume = sheet_thickness * np.pi * (self.beam.diameter_fwhm / 2) ** 2
        self.droplets = kwargs.get('droplets', False)
        if self.droplets:
            droplet_diameter = kwargs.get('droplet_diameter', 0.5e-6)
            self.water_temperature = kwargs.get('water_temperature', 298.0)
            self.droplet_volume = 4 / 3 * np.pi * (droplet_diameter / 2) ** 3
        self.gas = kwargs.get('gas_background', False)
        if self.gas:
            gas_type = kwargs.get('gas_type', 'he')
            gas_path_legnth = kwargs.get('gas_path', 1.0)
            gas_pressure = kwargs.get('gas_pressure', 101325.0)
            gas_temperature = kwargs.get('gas_temperature', 293.15)
            self.gas_background = get_gas_background(pad_geometry=self.pads,
                                                     beam=self.beam,
                                                     path_length=[0.0, gas_path_length],
                                                     gas_type=gas_type,
                                                     temperature=gas_temperature,
                                                     pressure=gas_pressure,
                                                     poisson=False)
        self.poisson = kwargs.get('poisson_noise', False)

    def get_data(self, frame_number=0):
        # Calculate the diffracted intensity of the molecule
        # I(q) = J_0 SA r_e^2 P(q) | sum_n f_n(q) sum_m exp(i q dot r_mn) |^2
        rotations = Rotation.random(num=self.n_particles).as_matrix()
        amplitudes = 0
        for r in rotations:
            for fs, rs in zip(self.grouped_fs, self.grouped_r_vecs):
                amplitudes += fs * self.simulation_engine.phase_factor_qrf(q=self.q_vecs,
                                                                           r=rs,
                                                                           R=r)
        intensity = np.abs(amplitudes) ** 2
        if self.sheets:
            intensity += water_scattering_factor_squared(q=self.q_mags,
                                                         temperature=self.water_temperature,
                                                         volume=self.sheet_volume)
        if self.droplets:
            intensity += water_scattering_factor_squared(q=self.q_mags,
                                                         temperature=self.water_temperature,
                                                         volume=self.droplet_volume)
        intensity *= self.f2phot
        if self.gas:
            intensity += self.gas_background
        if self.poisson:
            intensity = np.random.poisson(intensity)
        df = DataFrame(pad_geometry=self.pads,
                       beam=self.beam,
                       raw_data=intensity)
        return df


class Simulated(FrameGetter):

    def __init__(self, h5_path, **kwargs):
        super().__init__()
        self.file_list = glob(f'{h5_path}/*.h5')
        self.poisson = kwargs.get('poisson_noise', False)

    def get_data(self, frame_number=0):
        with h5py.File(self.file_list[frame_number], 'r') as hf:
            photon_energy = hf['beam/photon_energy'][()]
            detector_distance = hf['geometry/detector_distance'][()]
            intensity = hf['data'][:]
            if self.poisson:
                intensity = np.random.poisson(intensity)
            beam = Beam(photon_energy=photon_energy)
            pads = jungfrau4m_pad_geometry_list(detector_distance=detector_distance)
            df = DataFrame(pad_geometry=pads,
                           beam=beam,
                           raw_data=intensity)
        return df

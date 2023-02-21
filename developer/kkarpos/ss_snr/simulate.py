import numpy as np

from glob import glob
from reborn.dataframe import DataFrame
from reborn.detector import RadialProfiler
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

    Adapted from Roberto Alvarez's initial fluctuation scattering simulations.

    Arguments
    ---------
        pad_geometry (|PADGeometryList|): Detector geometry
        beam (|Beam|): X-ray beam
        denss_dat (|str|): The path to the denss .dat file.
        kwargs:
            experiment_id (str): Simulation identifier (default is 'simulation')
            run_id (int): Simulation run identifier (default is 0)
            n_frames (int): Number of simulations to run (default is 1)
            pdb (str): PDB id to simulate (this will setup crystal object for you) (Default is None)
            crystal (|CrystalStructure|): Crystal object (only use this if you need to do
                                          something with the crystal, i.e. enable molecular machine etc.)
            n_particles (int): Number of particles in the simulation. (default is 1)
            random_seed (int): Seed if reproducible random number are needed (default is None)
            jet (bool): Add solution background from gdvn jet (default is False)
                jet_thickness (float): Thickness of gdvn jet (default is 1e-6)
                water_temperature (float): Temperature of gdvn jet (default is 298.0 K)
            droplets (bool): Add solution background from droplet (default is False)
                droplet_diameter (bool): Diameter of water droplet (default is 0.5e-6)
                water_temperature (bool): Temperature of droplet (default is 298 K)
            gas_background (bool): Add gas background (default is False)
                gas_type (str): Type of gas background ('helium', 'he', or 'air'; default is 'he')
                gas_path (int): Length of full gas path (default is 1 m)
                gas_pressure (float): Simulated gas pressure (default 101325.0 Pa)
                gas_temperature (float): Simulated gas temperature (default is 293.15 K)
            poisson_noise (bool): Add Poisson noise to simulation (default is False)
            header_level (int): the number of header lines to skip in the denss .dat file (default is 0)
            n_radial_bins (int): The number of radial q bins to use when making radial profiles (default is 1000)
        """
    def __init__(self, pad_geometry, beam, denss_data, pdb, **kwargs):
        super().__init__()
        self.experiment_id = kwargs.get('experiment_id', 'simulation')
        self.run_id = kwargs.get('run_id', 0)
        self.n_frames = kwargs.get('n_frames', 1)
        self.denss_data = denss_data
        self.header = kwargs.get('header_level', 0)

        random_seed = kwargs.get('random_seed', None)

        self.scale_by = kwargs.get('scale_by', 1)

        if random_seed is not None:
            np.random.seed(random_seed)

        self.beam = beam
        self.fluence = self.beam.photon_number_fluence
        self.photon_energy = self.beam.photon_energy

        self.pads = pad_geometry
        self.q_vecs = self.pads.q_vecs(beam=self.beam)
        self.q_mags = self.pads.q_mags(beam=self.beam)
        self.f2phot = self.pads.f2phot(beam=self.beam)

        self.pdb_id = pdb 


        self.jet = kwargs.get('jet', True)
        if self.jet:
            jet_thickness = kwargs.get('jet_thickness', 1e-6)
            self.water_temperature = kwargs.get('water_temperature', 293.15)
            self.jet_volume = jet_thickness * np.pi * (self.beam.diameter_fwhm / 2) ** 2
            volume = self.jet_volume
        self.droplets = kwargs.get('droplets', False)
        if self.droplets:
            droplet_diameter = kwargs.get('droplet_diameter', 0.5e-6)
            self.water_temperature = kwargs.get('water_temperature', 298.0)
            self.droplet_volume = 4 / 3 * np.pi * (droplet_diameter / 2) ** 3
            volume = self.droplet_volume
        if (self.jet is False) and (self.droplets is False):
            raise ValueError("Pick a sample delivery method, 'jet' or 'droplet'")
        self.gas = kwargs.get('gas_background', False)
        # self.n_particles = kwargs.get('n_particles', None)  # 1,2,10,100
        self.concentration = kwargs.get('concentration', 10)  # mg/ml
        self.n_particles = self.get_n_proteins(concentration=self.concentration, 
                                                volume=volume, pdb_id=pdb)

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

        self.n_bins = kwargs.get('n_radial_bins', 1000)
        self.q_mags = self.pads.q_mags(beam=self.beam)
        self.profiler = RadialProfiler(beam=self.beam, pad_geometry=self.pads, 
                                        n_bins=self.n_bins, 
                                        q_range=np.array([np.min(self.q_mags), np.max(self.q_mags)]))

    def get_n_proteins(self, 
                        concentration,
                        volume, 
                        pdb_id:str, 
                        random=True):
        _xbeam = self.beam
        pdb = crystal.CrystalStructure(pdb_id, tempdir="tempdir")
        protein_number_density = concentration / pdb.molecule.get_molecular_weight()
        n_proteins = int(protein_number_density * volume)
        if random:
            n_proteins = int(np.random.normal(loc=n_proteins))
        return n_proteins


    def get_denss_data(self, filepath, header_level:int=0):
        r"""Loads data from denss
        """

        d = np.loadtxt(filepath, skiprows=header_level)
        q = d[:, 0] * 1e10
        amp = d[:, 1] * self.scale_by
        return q, amp

    def get_denss_pad_solution_intensity(self, denss_file,
                                            denss_header_level:int=0):


        #  make a pad_geometry copy so you don't mess with the original
        qmags, I = self.get_denss_data(denss_file, header_level=denss_header_level)
        F2 = np.interp(self.q_mags, qmags, I) * self.n_particles
        return F2

    def get_data(self, frame_number=0):
        # Calculate the diffracted intensity of the molecule
        # I(q) = J_0 SA r_e^2 P(q) | sum_n f_n(q) sum_m exp(i q dot r_mn) |^2

        intensity = self.get_denss_pad_solution_intensity(denss_file=self.denss_data, 
                                                            denss_header_level=self.header)

        # intensity = 0
        if self.jet:
            intensity += water_scattering_factor_squared(q=self.q_mags,
                                                         temperature=self.water_temperature,
                                                         volume=self.jet_volume)
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

    def get_radial(self, frame_number=0):
        return self.profiler.quickstats(self.get_data(frame_number).get_processed_data_flat())





























from time import time
import numpy as np
import pylab as plt
import reborn
import os
import matplotlib as mpl
import h5py
from scipy import constants as const
from reborn.viewers.qtviews.padviews import PADView2
from reborn.viewers.mplviews import view_pad_data
from reborn.target import crystal, atoms, molecule
from scipy.spatial.transform import Rotation
from reborn.simulate import solutions, clcore, gas
from reborn.detector import RadialProfiler
from reborn.detector import PADGeometry, PADGeometryList

class XRay_Diffraction:

    def __init__(self,
                    pad_geometry,
                    photon_energy: float=None,
                    # pulse_energy: float=None,
                    beam=None,
                    mask=None,
                    radial_profiler=None,
                    radial_q_range=None,
                    n_radial_bins=1000
                    ):

        # Ensure PADGeometryList type
        self.pad_geometry = reborn.detector.PADGeometryList(pad_geometry).copy()
        self.mask = mask
        self.photon_energy = photon_energy

        # build the beam
        if beam:
            self.beam = beam
        else:
            print("No beam class instance given, making one instead.")
            self.beam = reborn.source.Beam(photon_energy=self.photon_energy) #, pulse_energy=self.pulse_energy)
        self.photon_number_fluence = self.beam.photon_number_fluence

        # Get radial profiler info
        if radial_profiler:
            self.radial_profiler = radial_profiler
        else:
            print("No radial profile class instance given, making one instead.")
            if radial_q_range is None:
                qs = self.pad_geometry.q_mags(beam=self.beam)
                # print(np.min(qs), np.max(qs))
                self.radial_q_range = np.array([np.min(qs), np.max(qs)])#*1e10 #np.array([0, 3.2])*1e10 #np.array([np.min(qs), np.max(qs)])#*1e10
            else:
                self.radial_q_range = radial_q_range
            self.n_radial_bins = n_radial_bins
            self._initialize_radial_class()

        # constants
        self.r_e = const.value('classical electron radius') # meters
        self.kb = const.value('Boltzmann constant')  # J  K^-1
        self.na = const.value('Avogadro constant')  # mole^-1
        self.eV = const.value('electron volt')  # J

        self.molecules = {'N2': molecule.Molecule(coordinates=np.array([[0, 0, 0], [0, 0, 1.07e-10]]), atomic_numbers=[7, 7]),
                          'O2': molecule.Molecule(coordinates=np.array([[0, 0, 0], [0, 0, 1.21e-10]]), atomic_numbers=[8, 8]),
                          'He': molecule.Molecule(coordinates=np.array([[0, 0, 0]]), atomic_numbers=[2, 2])
                          }

    def get_gas_background(self,
                            gas_type: str,
                            iteration_steps: int=20,
                            nshots: int=1,
                            detector_distance: float=None,
                            path_length=None,
                            temperature: float=293.15,
                            pressure: float=101325.0,
                            poisson_noise=False,
                            verbose=False):

        # get simulation info
        if (detector_distance is not None) & (path_length is not None):
            raise ValueError("Please choose detector_distance or path_length, not both.")
        if detector_distance:
            print(f'path length: {detector_distance:.5f} meters')
            iter_list = np.linspace(1e-3, detector_distance, iteration_steps)
        if path_length:
            print(f'path length: {np.abs(path_length[0]) + np.abs(path_length[1]):.5f} meters')
            iter_list = np.linspace(path_length[0], path_length[1], iteration_steps)

        dx = iter_list[1] - iter_list[0]  # meters, step size of each iteration

        # calculate simulation parameters
        n_molecules = pressure * dx * np.pi * (self.beam.diameter_fwhm/2) ** 2 / (self.kb * temperature)
        alpha = self.r_e ** 2 * self.photon_number_fluence

        # make a pad_geometry copy so you don't mess with the original
        _pads = self.pad_geometry.copy()
        _xbeam = self.beam
        _xpe = self.beam.photon_energy
        gas_choices = ['N2', 'O2', 'He', 'air']
        if gas_type not in gas_choices:
            raise ValueError(f"Incorrect choice for gas! Current options are \n {gas_choices}")
        if gas_type =='air':
            pass
        else:
            _molecule = self.molecules[gas_type]

        # iterate from chamber length to detector
        I_total = 0
        count = 1
        for step in iter_list:
            if verbose:
                print(f"iteration: {count}/{iteration_steps}")
                print(f'\t step distance: \t {step:.4f} meters')
            # update the detector distance for this iteration

            for pad in _pads:
                pad.t_vec[2] = step
            t = time()
            # update parameters based on new pad geometry values
            q_mags = _pads.q_mags(beam=_xbeam)
            if verbose:
                print(f'\t q_mags: \t {time()-t:.3f} seconds')
            t = time()
            polarization = _pads.polarization_factors(beam=_xbeam)
            if verbose:
                print(f'\t polarization: \t {time()-t:.3f} seconds')
            t = time()
            solid_angles = _pads.solid_angles2()  # Approximate solid angles, computes faster
            if verbose:
                print(f'\t solid_angles: \t {time()-t:.3f} seconds')

            # update the scattering factors
            t = time()
            if gas_type == 'air':
                scatt = gas.air_intensity_profile(q_mags=q_mags, beam=_xbeam)
            else:
                scatt = gas.isotropic_gas_intensity_profile(molecule=_molecule, 
                                                            beam=_xbeam, 
                                                            q_mags=q_mags)
            if verbose:
                print(f'\t he_scatt: \t {time()-t:.3f} seconds')

            # Calculate new scattering intensity
            F2 = np.abs(scatt) ** 2 * n_molecules
            I = alpha * polarization * solid_angles * F2
            # sum the results
            I_total += I
            count += 1

        if poisson_noise is True:
            I_total = np.sum(np.random.poisson(lam=I_total, size=(nshots,) + np.shape(I_total)), axis=0).astype(np.double)

        # Split the concatenated data into a list of 2D arrays.
        I_split = _pads.split_data(I_total)

        return I_total #I_split

    def get_water_profile(self,
                            sample_path_length: float,
                            temperature: float,
                            sample_distance: float=None,
                            poisson=False,
                            nshots=1):

        #  make a pad_geometry copy so you don't mess with the original
        _pads = self.pad_geometry.copy()
        _xbeam = self.beam
        alpha = self.r_e ** 2 * self.photon_number_fluence

        volume = sample_path_length * np.pi * (_xbeam.diameter_fwhm / 2) ** 2
        water_total = solutions.water_scattering_factor_squared(q=_pads.q_mags(beam=_xbeam),
                                                                temperature=temperature,
                                                                volume=volume)
        I = alpha * _pads.polarization_factors(beam=_xbeam) * _pads.solid_angles2() * water_total
        return I #, water_scattering_params


    def get_atom_profile(self, atomic_number: int):

        r"""Calculates scattering intensity for single atom
            
            Arguments:
                atomic_number (int): As is sounds, the number of the desired atom
            Returns:
                Scattering Intensity [photon count]
        """
        #  make a pad_geometry copy so you don't mess with the original
        _pads = self.pad_geometry.copy()
        _xbeam = self.beam
        # constants
        alpha = self.r_e ** 2 * self.photon_number_fluence
        # define params
        q_mags = _pads.q_mags(beam=_xbeam)
        q_vecs = _pads.q_vecs(beam=_xbeam)
        polarization = _pads.polarization_factors(beam=_xbeam)
        solid_angles = _pads.solid_angles2()
        # calculate scattering factors for atom
        scatt = reborn.target.atoms.cmann_henke_scattering_factors(q_mags, atomic_number, beam=_xbeam)
        # calculate intensity
        I = alpha * polarization * solid_angles * np.abs(scatt) ** 2
        return I

    def get_temperature_scan(self,
                                sim_config:dict,
                                sample:str='water',
                                verbose=False,
                                poisson_noise=False):

        _pads = self.pad_geometry.copy()
        _xbeam = self.beam

        # check that the config has the right parameters for the desired sample
        if sample == 'water':
            config_keys = ['sample_path_length', 'temperature_range', 'detector_distance']
            for k in config_keys:
                if k not in sim_config:
                    err = f"""Simulation config keys are incorrect. 
                            Sample is water, but the config does not contain the following keys:
                            {config_keys}"""
                    raise ValueError(err)
        elif sample == 'gas':
            config_keys = ['gas_type', 'iteration_steps', 'path_length', 'pressure', 'temperature_range', 'detector_distance']
            for k in config_keys:
                if k not in sim_config:
                    err = f"""Simulation config keys are incorrect. 
                            Sample is gas, but the config does not contain the following keys:
                            {config_keys}"""
                    raise ValueError(err)

        temperature_range = sim_config['temperature_range']

        # initialize the radial array
        rads = np.zeros((len(temperature_range), self.n_radial_bins))
        t_count = 1
        # calculate the radial profile for each temperature
        for i, t in enumerate(temperature_range):
            if sample == 'water':
                profile, _ = self.get_water_profile(sample_path_length=sim_config['sample_path_length'],
                                                        temperature=t,
                                                        sample_distance=sim_config['detector_distance'],
                                                        poisson=poisson_noise) 
            elif sample == 'gas':
                profile = self.get_gas_background(gas_type=sim_config['gas_type'],
                                                        path_length=sim_config['sample_path_length'],
                                                        temperature=t,
                                                        pressure=sim_config['pressure'],
                                                        iteration_steps=sim_config['n_steps'],
                                                        poisson_noise=poisson_noise)
            # profile = _pads.concat_data(profile)
            # profile /= _pads.solid_angles() * _pads.polarization_factors(beam=self.beam)
            rad = self.get_mean_profile(profile)
            rads[i, :] = rad
            if verbose:
                print(f"Temperature {t_count}/{len(temp)} done!")
            t_count += 1
        return rads

    def get_temperature_differences(self,
                                    sim_config: dict):

        if sim_config["sample"] not in ['water', 'gas']:
            err = """sample_type incorrect. Current choices are limited to 'water' and 'gas'. 
                        To specify gas type, pass type into the sim_config."""
            raise ValueError(err)

        temp_range = sim_config['temperature_range']
        dt_dict = {}
        rads = self.get_temperature_scan(sim_config=sim_config, sample=sim_config["sample"])
        for i in np.flip(range(len(rads))):
            for j in range(i+1, len(rads)):
                if temp_range[j] > temp_range[i]:
                    diff = temp_range[j] - temp_range[i]
                    dt_dict[f"{np.abs(diff)}"] = diff
                else:
                    diff = temp_range[i] - temp_range[j]
                    dt_dict[f"{np.abs(diff)}"] = diff
        return rads, dt_dict


    def plot_temperature_scan(self,
                                sim_config: dict,
                                sample_type:str='water',
                                savefig=False,
                                savepath=None):

        if sample_type not in ['water', 'gas']:
            err = """sample_type incorrect. Current choices are limited to 'water' and 'gas'. 
                        To specify gas type, pass type into the sim_config."""
            raise ValueError(err)


        rads = self.get_temperature_scan(sim_config=sim_config, sample=sample_type)
        temp_range = sim_config['temperature_range']
        q_range = np.linspace(self.radial_q_range[0]*1e-10, self.radial_q_range[1]*1e-10, self.n_radial_bins)

        mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors', np.flip(['red', 'orange', 'yellow', 'green', 'blue' ]))
        vmin, vmax = (0, max(temp_range)-min(temp_range))
        step = 0.1
        Z = [[0,0],[0,0]]
        tempdiff_dict = {}

        plt.figure(figsize=(12,5))
        levels = np.arange(vmin, vmax+step, step)
        CS3 = plt.contourf(Z, levels, cmap=mymap)
        plt.clf()
        for i in np.flip(range(len(rads))): 
            for j in range(i+1, len(rads)):
                if temp_range[j]>temp_range[i]:
                    del_t = np.abs(temp_range[j]-temp_range[i])
                    plt.plot(q_range,
                             rads[j]-rads[i], 
                             label=str(temp_range[j] - temp_range[i]),
                             c=mymap((temp_range[j] - temp_range[i])/vmax))
                    tempdiff_dict['{}'.format(del_t)] = rads[j] - rads[i]
                else:
                    del_t = int(np.abs(temp_range[i]-temp_range[j]))
                    plt.plot(q_range, 
                             rads[i]-rads[j], 
                             label=str(temp_range[i] - temp_range[j]), 
                             c=mymap((temp_range[j] - temp_range[i])/vmax))
                    tempdiff_dict['{}'.format(del_t)] = rads[i]-rads[j]
        diffarray= np.array(list(tempdiff_dict.items()))
        plt.colorbar(CS3);
        if sample_type == 'gas':
            s = sim_config['gas_type']
        else:
            s = sample_type
        plt.title(f'Simulated {s} Difference Profiles', fontsize=16);
        plt.xlabel('q [$\AA^{-1}$]', fontsize=14);
        # plt.xlim((1.4, 3))
        if savefig:
            plt.savefig(savepath)
        else:
            plt.show()


    def get_molecular_weight_from_pbd_file(self, filepath):
        r"""
        Returns the molecular weight in SI units (kg).
        """
        if os.path.splitext(filepath)[-1] != ".pdb":
            raise ValueError("filepath is not a .pdb file!")
        pdb_dict = crystal.pdb_to_dict(filepath)
        atomic_numbers = atoms.atomic_symbols_to_numbers(pdb_dict['atomic_symbols'])
        return np.sum(atoms.atomic_weights[atomic_numbers])

    def _get_denss_data(self, filepath):
        r"""Loads data from denss
        """
        d = np.loadtxt(filepath)
        q = d[:, 0] * 1e10
        diff = d[:, 1]
        err = d[:, 2]
        return q, diff, err


    def denss_scattering_factor_squared(self, 
                                            q, 
                                            denss_file, 
                                            pdb_id:str=None, 
                                            protein_concentration:float=None,
                                            tempdir:str=None):
        r"""Get the scattering factor squared from a file produced with DENSS
            Returns (array): scattering factor squared
        """
        qmag, I, err = self._get_denss_data(denss_file)
        F2 = np.interp(q, qmag, I)
        # if (pdb_id is not None) & (protein_concentration is not None):
        #     if tempdir is None:
        #         x = os.path.join(os.sep, *denss_file.split('/')[1:-1], 'tempdir')
        #         os.makedirs(os.path.join(x), exist_ok=True)
            # pdb = crystal.CrystalStructure(pdb_id, tempdir=tempdir)
            # F2 = F2 * protein_concentration * pdb.molecule.get_molecular_weight()
        return F2

    def get_n_proteins(self, 
                        protein_concentration,
                        path_length:float=1.0, 
                        pdb_id:str='1JFP', 
                        tempdir:str=None,
                        random=True):
        _xbeam = self.beam
        volume = path_length * np.pi * (_xbeam.diameter_fwhm / 2) ** 2
        pdb = crystal.CrystalStructure(pdb_id, tempdir=tempdir)
        protein_number_density = protein_concentration / pdb.molecule.get_molecular_weight()
        n_proteins = int(protein_number_density * volume)
        if random:
            n_proteins = int(np.random.normal(loc=n_proteins))
        return n_proteins

    def get_denss_pad_solution_intensity(self, 
                                            denss_file,
                                            path_length:float=1,
                                            nshots=1,
                                            pdb_id:str=None, 
                                            protein_concentration:float=None,
                                            tempdir:str=None,
                                            poisson=False):

        #  make a pad_geometry copy so you don't mess with the original
        _pads = self.pad_geometry.copy()
        _xbeam = self.beam
        q_mags = _pads.q_mags(beam=_xbeam)

        # constants
        # volume = path_length * np.pi * (_xbeam.diameter_fwhm / 2) ** 2
        alpha = self.r_e ** 2 * self.photon_number_fluence

        qmags, diff, err = self._get_denss_data(denss_file)
        scatt = self.denss_scattering_factor_squared(q_mags, denss_file, pdb_id, protein_concentration, tempdir)
        if tempdir is None:
            x = os.path.join(os.sep, *denss_file.split('/')[1:-1], 'tempdir')
            os.makedirs(os.path.join(x), exist_ok=True)
        # pdb = crystal.CrystalStructure(pdb_id, tempdir=tempdir)
        # protein_number_density = protein_concentration / pdb.molecule.get_molecular_weight()
        # n_proteins = int(protein_number_density * volume)
        n_proteins = self.get_n_proteins(protein_concentration, path_length, pdb_id, tempdir)
        F2 = scatt * n_proteins
        polarization = _pads.polarization_factors(beam=_xbeam)
        solid_angles = _pads.solid_angles1()
        I = alpha * solid_angles * polarization * F2
        if poisson is True:
            I = np.sum(np.random.poisson(lam=I, size=(nshots,) + np.shape(I)), axis=0).astype(np.double)
        return I


    def get_snr_diffprofile(self, norm_diff, I1, I2, nshots):
        return norm_diff * np.sqrt(nshots / (I1 + I2))


    def get_pdb_profile(self,
                            pdb_id: str,
                            protein_concentration: float,
                            jet_width: float=None,
                            drop_radius: float=None,
                            rotate_molecule: bool=True,
                            sample_delivery_method: str="jet",
                            tempdir: str=None,
                            poisson=False,
                            verbose=False):

        #  make a pad_geometry copy so you don't mess with the original
        _pads = self.pad_geometry.copy()
        _xbeam = self.beam
        _xpe = self.beam.photon_energy

        # constants
        alpha = self.r_e ** 2 * self.photon_number_fluence

        # define params
        q_mags = _pads.q_mags(beam=_xbeam)
        q_vecs = _pads.q_vecs(beam=_xbeam)
        polarization = _pads.polarization_factors(beam=_xbeam)
        solid_angles = _pads.solid_angles1()

        # get pdb info
        cryst = crystal.CrystalStructure(pdb_id, tempdir=tempdir)
        protein_number_density = protein_concentration/cryst.molecule.get_molecular_weight()
        if (sample_delivery_method == "jet") and (jet_width):
            volume = jet_width * np.pi * (_xbeam.diameter_fwhm / 2) ** 2
        elif (sample_delivery_method == "drop") and (drop_radius):
            volume = 4/3 * np.pi * drop_radius**3
        else: 
            raise ValueError("If sample_delivery_method is 'jet', a jet_width must be given. Similarily for 'drop' and drop_radius.")

        n_proteins = int(protein_number_density * volume)
        r_vecs = cryst.molecule.coordinates  # Atomic coordinates of the asymmetric unit
        r_vecs -= np.mean(r_vecs, axis=0)  # Roughly center the molecule
        atomic_numbers = cryst.molecule.atomic_numbers
        simcore = clcore.ClCore()

        t = time()
        uniq_z = np.unique(atomic_numbers)
        grouped_r_vecs = []
        grouped_fs = []
        for z in uniq_z:
            subr = np.squeeze(r_vecs[np.where(atomic_numbers == z), :])
            grouped_r_vecs.append(subr)
            grouped_fs.append(atoms.hubbel_henke_scattering_factors(q_mags=q_mags, 
                                                                    photon_energy=_xpe,
                                                                    atomic_number=z))
        if verbose:
            print(f"\t\t Scattering factors calculated: {time()-t:.2f} seconds")

        t = time()
        count = 1
        # rotate the molecule randomly
        #FIXME: get rotational average here
        if rotate_molecule:
            R = Rotation.random().as_matrix()
        amps = 0
        for j in range(len(grouped_fs)):
            f = grouped_fs[j]
            r = grouped_r_vecs[j]
            a = simcore.phase_factor_qrf(q_vecs, r, R=R)
            amps += a*f
        F2 = np.abs(amps) ** 2 * n_proteins
        I = alpha * solid_angles * polarization * F2
        if poisson is True:
            I = np.random.poisson(I).astype(np.double)
        if verbose:
            print(f"\t\t Pad intensities calculated: {time()-t:.2f} seconds")
        return I



    """
    Radial Stuff for convinience
    """
    def _initialize_radial_class(self):
        _xbeam = self.beam
        _pads = self.pad_geometry.copy()
        self.profiler = RadialProfiler(beam=_xbeam,
                                    pad_geometry=_pads,
                                    n_bins=self.n_radial_bins,
                                    q_range=self.radial_q_range,
                                    mask=self.mask)
        return self.profiler

    def get_profile_statistic(self, data, statistic):
        return self.profiler.get_profile_statistic(data, statistic=statistic)

    def get_sdev_profile(self, data):
        return self.profiler.get_sdev_profile(data)

    def get_counts_profile(self, data):
        return self.profiler.get_counts_profile(data)

    def get_mean_profile(self, data):
        return self.profiler.get_mean_profile(data)

    def get_sum_profile(self, data):
        return self.profiler.get_sum_profile(data)



class SimulationFrameGetter(reborn.fileio.getters.FrameGetter):

    def __init__(self, data, beams, pad_geometry, mask=None):
        super().__init__()
        self.pad_geometry = pad_geometry
        self.mask = mask
        self.n_frames = len(data)
        shots = [i for i in range(self.n_frames)]
        self.frames = dict(zip(shots, data))
        self.beams = dict(zip(shots, beams))

    def get_data(self, frame_number=0):
        current_frame = self.frames[frame_number]
        pads_data = current_frame
        df = reborn.dataframe.DataFrame()
        df.set_frame_id(frame_number)
        df.set_pad_geometry(self.pad_geometry)
        df.set_raw_data(pads_data)
        df.set_beam(self.beams[frame_number])
        if self.mask is not None:
            df.set_mask(self.mask)
        return df


class h5Tools:

    def __init__(self, h5_name:str, outpath:str):
        self.h5_name = h5_name
        self.outpath = outpath

        try:
            os.makedirs(self.outpath)
        except FileExistsError:
            pass


    def write_data_as_h5(self, *args, group_name='default', verbose=True):
        r'''
        1) h5_name is the file output, include the full path here as a string
        2) group_name is the name of the top level h5 folder you're creating, must be a string
        3) args expects a tuple in the format ('dataset_name', data),
            where 'dataset_name' is a string and data is the actual data
        !!! Note that the order matters. Start with the keys, then the h5_name and group_name !!!
        '''
        try:
            h5 = h5py.File(self.h5_name, 'a')
        except ValueError:
            h5 = h5py.File(self.h5_name, 'w')
        try:
            group = h5.create_group(group_name)
        except ValueError:
            group = h5['{}'.format(group_name)]

        for arg in args:
            group.create_dataset(arg[0], data=arg[1])

        h5.close()
        if verbose is True:
            print('Created the {} group!'.format(group_name))
            print('Closed {}!'.format(self.h5_name))














































from time import time
import numpy as np
import pylab as plt
import reborn
import os
from reborn.detector import RadialProfiler
import matplotlib as mpl
from .diffraction_simulation import XRay_Diffraction


class RadialTools:

    def __init__(self,
                    pad_geometry,
                    beam,
                    n_radial_bins=1000,
                    radial_q_range=np.array([0, 3.2])*1e10,
                    mask=None,
                    xray_diffraction=None):

        self.pad_geometry = pad_geometry.copy()
        self.beam = beam
        self.n_radial_bins = n_radial_bins
        self.radial_q_range = radial_q_range
        self.mask = mask
        if mask is None:
            print('No mask given, self.mask will return None.')
        self.xray_diffraction = xray_diffraction

        self.profiler = RadialProfiler(beam=self.beam,
                                        pad_geometry=self.pad_geometry,
                                        n_bins=self.n_radial_bins,
                                        q_range=self.radial_q_range,
                                        mask=self.mask)


    # wrapper functions for easy in-class use.
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

    def _simulate_temperature_scan(self,
                                    sim_config:dict,
                                    sample:str='water',
                                    verbose=False,
                                    poisson_noise=False):

        # check that the XRayDiffraction class is initialized
        if self.xray_diffraction is None:
            self.xray_diffraction = XRay_Diffraction(beam=self.beam, pad_geometry=self.pad_geometry)

        _pads = self.pad_geometry.copy()
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
                profile, _ = self.xray_diffraction.get_water_profile(sample_path_length=sim_config['sample_path_length'],
                                                                temperature=t,
                                                                sample_distance=sim_config['detector_distance'],
                                                                poisson=poisson_noise) 
            elif sample == 'gas':
                profile = self.xray_diffraction.get_gas_background(gas_type=sim_config['gas_type'],
                                                                    path_length=sim_config['sample_path_length'],
                                                                    temperature=t,
                                                                    pressure=sim_config['pressure'],
                                                                    iteration_steps=sim_config['n_steps'],
                                                                    poisson_noise=poisson_noise)
            profile = _pads.concat_data(profile)
            profile /= _pads.solid_angles() * _pads.polarization_factors(beam=self.beam)
            rad = self.get_mean_profile(profile)
            rads[i, :] = rad
            if verbose:
                print(f"Temperature {t_count}/{len(temp)} done!")
            t_count += 1
        return rads



    def plot_temperature_scan(self,
                                sim_config: dict,
                                sample_type:str='water',
                                savefig=False,
                                savepath=None):

        if sample_type not in ['water', 'gas']:
            err = """sample_type incorrect. Current choices are limited to 'water' and 'gas'. 
                        To specify gas type, pass type into the sim_config."""
            raise ValueError(err)


        rads = self._simulate_temperature_scan(sim_config=sim_config, sample=sample_type)
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
                             c=mymap((temp_range[j]-temp_range[i])/vmax))
                    tempdiff_dict['{}'.format(del_t)] = rads[j]-rads[i]
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
















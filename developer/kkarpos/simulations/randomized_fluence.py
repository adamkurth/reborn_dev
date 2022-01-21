from time import time
import numpy as np
import pylab as plt
import reborn
import os, sys
import json
from scipy import constants as const
import reborn

from tools.diffraction_simulation import XRay_Diffraction, SimulationFrameGetter, h5Tools
from tools.pad_geom_tools import PADTools

tik = time()

eV = const.value('electron volt')  # J
np.random.seed(0)


# flip this to True to show histograms of the randomized data
show_plots = False
save_npz = True

"""
=================================
Set up the simulation parameters
=================================
"""

config = {'n_steps': 20, # number of helium chamber spatial steps
            'bin_pixels': 10, # pixel bin size
            'n_shots': 2, #1000, # number of exposure events to integrate
            'detector_distance': 0.056, # m
            'gas_pathlength': [-0.02, 0.03], # m, [len of gas before interaction region, len of gas after]
            'kapton_window_diameter': 0.31115, # m, 12.25 inches. Any pixels outside of this diameter are masks 
            'nominal_photon_energy': 9500, # the nominal photon energy in eV, will add noise around this value.
            'nominal_sample_path_length': 5, # um, nominal sampe thickness, will add noise around this value
            'temperature': 293.15, # K
            'pressure': 101325.0, # Pa = 1 atm
            'histogram_bins': 5,
            'photon_energy_random_scale': 1,
            'sample_thickness_random_scale': 1/6
            }

outpath = '/Volumes/GoogleDrive/Shared drives/Ph.D. Work/Projects/random/'
np_save = outpath + 'data'

"""
=================================
Make the randomized data
=================================
"""

# make a list of randomized photon energies
photon_energy_eV = np.random.normal(loc=config['nominal_photon_energy'], 
                                    scale=config['photon_energy_random_scale'],
                                    size=config['n_shots'])

# convert list to Joules (reborn is in SI)
photon_energy_J = photon_energy_eV * eV

if show_plots:
    # makes a histogram of photon energy values in keV
    fig, ax = plt.subplots(1,2, figsize=(12, 5), tight_layout=True)
    ax[0].hist(photon_energy_eV/1000, bins=config['histogram_bins'])
    ax[0].set_xlabel('Photon Energy [keV]')
    ax[0].set_title('Randomized Photon Energy - Normally Distributed')



# make a list of randomized sample thicknesses
sample_thickness_um = np.random.normal(loc=config['nominal_sample_path_length'], 
                                        scale=config['sample_thickness_random_scale'], 
                                        size=config['n_shots'])

# convert list to meters (reborn is in SI)
sample_thickness_m = sample_thickness_um * 1e-6

if show_plots:
    # makes a histogram of sample thicknesses in meters
    ax[1].hist(sample_thickness_um, bins=config['histogram_bins'])
    ax[1].set_xlabel('Sample Thickness [um]')
    ax[1].set_title('Randomized Sample Thickness - Normally Distributed')


if show_plots:
    # show the histogram plots
    fig.suptitle(f"Randomized Data -- {config['n_shots']} Total Shots")
    plt.show()



"""
=================================
Simulate the 2D patterns
=================================
"""

pads = reborn.detector.mpccd_pad_geometry_list(detector_distance=config['detector_distance'])
pads.binned(10)
# make multiple beam instances for each photon energy
beams = [reborn.source.Beam(photon_energy=s) for s in photon_energy_J]

data = [] #np.zeros((config['n_shots'], pads.zeros().shape[0]))

for s in range(config['n_shots']):
    xrd = XRay_Diffraction(beam=beams[s], pad_geometry=pads)

    I_total = pads.zeros()

    # simulate the water profile
    I_water = xrd.get_water_profile(sample_path_length=sample_thickness_m[s],
                                                temperature=config['temperature'],
                                                poisson=True) # returns flattened array
    # add to the total 2D pattern
    I_total += I_water

    # simulate a helium enviroment
    I_helium = xrd.get_gas_background(gas_type='He',
                                                path_length=config['gas_pathlength'],
                                                temperature=config['temperature'],
                                                pressure=config['pressure'],
                                                iteration_steps=5,
                                                poisson_noise=True) # returns flattened array
    # add to the total 2D pattern
    I_total += I_helium

    data.append(I_total)




"""
=================================
Display the data
=================================
"""

# display the pattern
fg = SimulationFrameGetter(data=data, pad_geometry=pads, beams=beams)
fg.view()


"""
=================================
Save the data
=================================
"""


if save_npz:
    np.savez(np_save, data=data)
    with open(np_save+'_config.txt','w') as out:
        json.dump(config, out)





















import numpy as np
import pylab as plt
import pandas as pd
import reborn
import os, sys
import time

from simulate import Simulator
from reborn.detector import rayonix_mx340_xfel_pad_geometry_list
from reborn.source import Beam
from reborn.const import eV
from reborn.viewers.qtviews import PADView


# Define the file paths to the DENSS data
# The first three here are rhodopsin in vacuum
diff_v = "denss_data/rhodopsin_in_vacuum/1jfp_diff.dat"
s1_v = "denss_data/rhodopsin_in_vacuum/1jfp_pdb_rho.dat"
s2_v = "denss_data/rhodopsin_in_vacuum/1jfp_modified_pdb_rho.dat"

# The next three are rhodopsin with and without retinal
# diff_s = "denss_data/rhodopsin_in_solution/sim_1-18.3_light_0.2ps-0.0ps_diff.dat"
# s1_s = "denss_data/rhodopsin_in_solution/sim_1-18.3_light_0.0ps.dat"
# s2_s = "denss_data/rhodopsin_in_solution/sim_1-18.3_light_0.2ps.dat"

# These three are the cis vs trans rhodopsin profiles
diff_s = "denss_data/cis_v_trans/sim_1-18.3_light_adjusted_0.2-0.0ps_diff.pdb2sas.dat"
s1_s = "denss_data/cis_v_trans/sim_1-18.3_light_0.0ps_adjusted.pdb2sas.dat"
s2_s = "denss_data/cis_v_trans/sim_1-18.3_light_0.2ps_adjusted.pdb2sas.dat"


# phytochrome
# s2_s = "denss_data/phytochrome/phytochrome_conf1_2vea_pdb.pdb2mrc2sas.dat"
# s1_s = "denss_data/phytochrome/phytochrome_conf2_3zq5_pdb.pdb2mrc2sas.dat"
# diff_s = 


in_vacuum = False           # Flip if you want to use the in-vacuum profiles
in_solution = True          # Set to True if you want to use the in-solution profiles
show_incoming_data = False  # Displays the full radial profiles in the .dat files
compare_diffs = False       # Compares the difference profiles calculated here vs the ones in the .dat files
show_det = False            # Shows the 2D detector after the simulation
plot_snr = True            # Plots the SNR 
print_beam_params = True    # Prints the parameters from the reborn beam class to the terminal

if in_vacuum:
    s1 = s1_v
    s2 = s2_v
    header_level = 0
elif in_solution:
    s1 = s1_s
    s2 = s2_s
    header_level = 1

def shannon_s(s=2, d=100):
    # returns shannon sampling ratio s in units of d
    return 2*np.pi / (s*d)

s = 2 # shannon sampling ratio
dmax = 100 #249 # A
# dmax = 249 # A

config = {'detector_distance': 1, # m
            'photon_energy': 8e3 * eV, 
            'sample_delivery_method': "jet",
            'jet_thickness': 300e-6, #5e-6, # m
            'droplet_diameter': 40e-6,
            'n_shots': np.array([1e3, 1e4, 1e5, 1e6]), # Only do 4 at a time or the plot looks screwy
            'helium': False,    # Flip this if you want to add helium into the mix
            'pdb_id':  "1JFP", #"2VEA"
            'concentration': 10, # mg/ml, the sample concentration
            'binned_pixels': 10, # The detector pixel binning, 1 is no binning
            'random_seed': 1,    # The random seed for Poisson noise
            'header_level': header_level, # DENSS file sometimes have headers
            'scale_by_1': 1,       # A manual scale amount applied to the DENSS profiles
            'scale_by_2': 1, #.062, 
            'beamstop_diameter': 0,   # m,
            'poisson_noise': False,

            }

pulse_energy = config['photon_energy'] * 1e8

dv = np.loadtxt(diff_v)
vs1 = np.loadtxt(s1_v)
vs2 = np.loadtxt(s2_v)
ds = np.loadtxt(diff_s, skiprows=0)
ss1 = np.loadtxt(s1, skiprows=1)
ss2 = np.loadtxt(s2, skiprows=1)
# ss1[:,1] *= 1.062



if show_incoming_data:
    # # Load the data

    # If you want to see and compare the profiles in solution wrt the profiles in vacuum
    fig, ax = plt.subplots(1,1, tight_layout=True)
    # ax[0].plot(vs1[:,0], vs1[:,1], label='state 1')
    # ax[0].plot(vs2[:,0], vs2[:,1], label='state 2')
    ax.plot(ss1[:,0], ss1[:,1])
    ax.plot(ss2[:,0], ss2[:,1])
    ax.plot(ss1[:,0], ss2[:,1]-ss1[:,1])
    ax.set_title('in solution')
    fig.legend()
    plt.show()

# Set up the detector and beam
pads = rayonix_mx340_xfel_pad_geometry_list(detector_distance=config['detector_distance'])
pads = pads.binned(config['binned_pixels'])
beam = Beam(pulse_energy=pulse_energy)#, 
config['n_radial_bins'] = int(np.max(pads.q_mags(beam))*1e-10/shannon_s(s=s, d=dmax))
qrange = np.linspace(0, np.max(pads.q_mags(beam)), config['n_radial_bins']) * 1e-10



# Define the sample delivery method
if config['sample_delivery_method'] == 'jet':
    jet = True
    jet_thickness = config['jet_thickness']
    droplets = False
    droplet_diameter = None
elif config['sample_delivery_method'] == 'droplets':
    jet = False
    droplets = True
    jet_thickness = None
    droplet_diameter = config['droplet_diameter']

# Set up the simulator class for the two states
fg = Simulator(pad_geometry=pads, beam=beam, denss_data=s1, pdb=config['pdb_id'],
                    n_radial_bins=config['n_radial_bins'], poisson_noise=config['poisson_noise'],
                    jet=jet, jet_thickness=jet_thickness, beamstop_diameter=config['beamstop_diameter'],
                    droplets=droplets, droplet_diameter=droplet_diameter, scale_by=config['scale_by_1'],
                    random_seed=config['random_seed'], header_level=config['header_level'])

fg_s2 = Simulator(pad_geometry=pads, beam=beam, denss_data=s2, pdb=config['pdb_id'],
                    n_radial_bins=config['n_radial_bins'], poisson_noise=config['poisson_noise'],
                    jet=jet, jet_thickness=jet_thickness, beamstop_diameter=config['beamstop_diameter'],
                    droplets=droplets, droplet_diameter=droplet_diameter, scale_by=config['scale_by_2'],
                    random_seed=config['random_seed'], header_level=config['header_level'])

# fig, ax = plt.subplots(2,1)
# ax[0].imshow(fg.pads[0].reshape(fg.f2phot), aspect='auto', interpolation='none')
# ax[1].imshow(fg_s2.pads[0].reshape(fg_s2.f2phot), aspect='auto', interpolation='none')
# plt.show()


if print_beam_params:
    print(f"\n\nBeam Parameters")
    print(f"---------------\n")
    print(f"N Photons Per Pulse: \t {fg.beam.n_photons:0.3e}")
    print(f"Pulse Energy: \t\t {fg.beam.pulse_energy} J")
    print(f"Pulse Fluence: \t\t {fg.beam.energy_fluence:.3e} J/m^2")
    print(f"Wavelength: \t\t {fg.beam.wavelength * 1e9:0.3f} nm")
    print(f"Photon Number Fluence: \t {fg.beam.photon_number_fluence:0.3e} photons/m^2")

# diff = fg.get_data(0) - fg_s2.get_data(0)


# Grab the radial profiles, no weights
rad_s1 = fg.get_radial(0)
rad_s2 = fg_s2.get_radial(0)
rad_s1_w = fg.get_radial(0, weights=fg.f2phot)
rad_s2_w = fg_s2.get_radial(0, weights=fg_s2.f2phot)

# Gather the quantities needed for the SNR calculations
# mean_diff = rad_s1['mean'] - rad_s2['mean'] # Standard state 1 - state 2 mean difference
sum_diff = rad_s1['sum'] - rad_s2['sum']    # The differences of the summed profiles
prof_sum = rad_s1['sum'] + rad_s2['sum']    # The total state 1 + state 2 profile

mean_diff = rad_s1_w['mean'] - rad_s2_w['mean']

tdiff = ss1[:,1]-ss2[:,1]
tqr = ss1[:,0]
tdiff *= np.max(np.abs(mean_diff)) / np.max(np.abs(tdiff))

if compare_diffs:

    md = mean_diff.copy()
    tdiff = ss1[:,1]-ss2[:,1]
    tdiff /= np.max(np.abs(tdiff))
    md /= np.max(np.abs(md))

    # If requested, will compare the difference profiles calculated here against Tom's
    fig, ax = plt.subplots()
    ax.plot(ss1[:,0], tdiff, label='toms')
    ax.plot(qrange, md, label='mine')
    fig.legend()
    plt.show()



def snr(difference, data, n_shots):
    r""" Calculates the signal to noise ratio of a difference profile as
            SNR = d|F(q)|^2 / sqrt(sum(I1+I2) / N)

            Args:
                difference (list): The calculated difference of the sum radials
                data (list): List containing the sum of the profiles used for the differences
                n_shots (int): Number of x-ray exposure events contributing to the pattern
            Returns:
                (list): The signal to noise ratio
    """
    return difference / np.sqrt(data / n_shots)

err_diff = [{'err': mean_diff / snr(sum_diff, prof_sum, n),
                'n_shots': n} for n in config['n_shots']]

# Convert the sample path length to microns
if config['sample_delivery_method'] == "jet":
    sample_pl_in_um = config['jet_thickness'] * 1e6
elif config['sample_delivery_method'] == "droplets":
    sample_pl_in_um = config['droplet_diameter'] * 1e6

if config['helium']:
    medium = "Helium"
else:
    medium = "Vacuum"
title_sem = f"""Mean Difference Profile,  N Shots Comparison
            Detector Distance = {config['detector_distance']}m, Photon Energy = {config['photon_energy']/(1000 * eV)} keV
            Sample Path Length: {sample_pl_in_um}$\mu$m, Medium: {medium}
        """

if show_det:
    pv = PADView(frame_getter=fg)
    pv.start()

if plot_snr:
    plt.figure(figsize=(15,8))
    lim = 0
    colors = [0.2, 0.5, 0.8] 
    count = 0
    for i in err_diff[:-1]:
        fill = (mean_diff + err_diff[count]['err'], mean_diff - err_diff[count]['err'])
        plt.fill_between(qrange, fill[0], fill[1], 
                              color='gray', edgecolor='gray', alpha=colors[count],
                              label=f"Error for {int(err_diff[count]['n_shots']):.0e} shots")
        count += 1
    fill = (mean_diff + err_diff[-1]['err'], mean_diff - err_diff[-1]['err'])
    plt.fill_between(qrange, fill[0], fill[1], 
                          color='lightsteelblue', edgecolor='lightsteelblue', alpha=0.8,
                          label=f"Error for {int(err_diff[-1]['n_shots']):.0e} shots")

    plt.plot(qrange, mean_diff, '--', color='black', label='Difference Profile')
    plt.plot(tqr, tdiff, '--', color='orange', label='Ground Truth (scaled)')
    plt.xlim(qrange[0], qrange[-1])
    plt.ylim(-np.max(mean_diff)*(1-0.9), np.max(mean_diff)*(1+0.35))
    plt.xlabel(r"q = 4 $\pi$ $\sin\theta$ / $\lambda$  $[\AA^{-1}]$", fontsize=14)
    plt.ylabel("Mean Difference Profile, I(q)", fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(alpha=0.3)
    plt.title(title_sem, fontsize=12)
    plt.show()












































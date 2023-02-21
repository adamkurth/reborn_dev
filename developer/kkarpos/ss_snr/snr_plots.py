import numpy as np
import pylab as plt
import pandas as pd
import reborn
import os, sys

from simulate import Simulator
from reborn.detector import rayonix_mx340_xfel_pad_geometry_list
from reborn.source import Beam
from reborn.const import eV
from reborn.viewers.qtviews import PADView



diff_v = "denss_data/rhodopsin_in_vacuum/1jfp_diff.dat"
s1_v = "denss_data/rhodopsin_in_vacuum/1jfp_pdb_rho.dat"
s2_v = "denss_data/rhodopsin_in_vacuum/1jfp_modified_pdb_rho.dat"
diff_s = "denss_data/rhodopsin_in_solution/sim_1-18.3_light_0.2ps-0.0ps_diff.dat"
s1_s = "denss_data/rhodopsin_in_solution/sim_1-18.3_light_0.0ps.dat"
s2_s = "denss_data/rhodopsin_in_solution/sim_1-18.3_light_0.2ps.dat"

in_vacuum = False
in_solution = True
show_incoming_data = False
compare_diffs = True
show_det = False
plot_snr = True

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
rhod_dmax = 100 # A

config = {'detector_distance': 1.5, # m
            'photon_energy': 10e3 * eV, #8e3 * eV,
            'n_radial_bins': int(3.2/shannon_s(s=s, d=rhod_dmax)),
            'sample_delivery_method': "jet",
            'jet_thickness': 6e-6, # m
            'droplet_diameter': 60e-6,
            'n_shots': np.array([1, 100, 1000, 5000]),
            'helium': False,
            'pdb_id': "1JFP",
            'concentration': 10, # mg/ml
            'binned_pixels': 10,
            'random_seed': 0,
            'header_level': header_level,
            'scale_by': 7.34e7,
            }


dv = np.loadtxt(diff_v)
vs1 = np.loadtxt(s1_v)
vs2 = np.loadtxt(s2_v)
ds = np.loadtxt(diff_s, skiprows=0)
ss1 = np.loadtxt(s1, skiprows=1)
ss2 = np.loadtxt(s2, skiprows=1)

if show_incoming_data:
    fig, ax = plt.subplots(2,1)
    ax[0].plot(vs1[:,0], vs1[:,1], label='state 1')
    ax[0].plot(vs2[:,0], vs2[:,1], label='state 2')
    ax[1].plot(ss1[:,0], ss1[:,1] * 7.34e7 , label='state 1')
    ax[1].plot(ss2[:,0], ss2[:,1] * 7.34e7 , label='state 2')
    ax[0].set_title('in vacuum')
    ax[1].set_title('in solution')
    fig.legend()

    plt.show()
    sys.exit()


pads = rayonix_mx340_xfel_pad_geometry_list(detector_distance=config['detector_distance'])
pads = pads.binned(config['binned_pixels'])
beam = Beam(photon_energy=config['photon_energy'])
qrange = np.linspace(0, np.max(pads.q_mags(beam)), config['n_radial_bins']) * 1e-10
# qrange = np.linspace(0, 3.5, config['n_radial_bins']) * 1e-10


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

fg = Simulator(pad_geometry=pads, beam=beam, denss_data=s1, pdb=config['pdb_id'],
                    n_radial_bins=config['n_radial_bins'], 
                    jet=jet, jet_thickness=jet_thickness,
                    droplets=droplets, droplet_diameter=droplet_diameter, scale_by=config['scale_by'],
                    random_seed=config['random_seed'], header_level=config['header_level'])
fg_s2 = Simulator(pad_geometry=pads, beam=beam, denss_data=s2, pdb=config['pdb_id'],
                    n_radial_bins=config['n_radial_bins'], 
                    jet=jet, jet_thickness=jet_thickness,
                    droplets=droplets, droplet_diameter=droplet_diameter, scale_by=config['scale_by'],
                    random_seed=config['random_seed'], header_level=config['header_level'])


rad_s1 = fg.get_radial(0)
rad_s2 = fg_s2.get_radial(0)

mean_diff = rad_s1['mean'] - rad_s2['mean']
sum_diff = rad_s1['sum'] - rad_s2['sum']
prof_sum = rad_s1['sum'] + rad_s2['sum']

if compare_diffs:
    fig, ax = plt.subplots()
    ax.plot(qrange, mean_diff, label='mine')
    ax.plot(ds[:,0], ds[:,1], label='toms')
    fig.legend()
    plt.show()



def snr(difference, data, n_shots):
    return difference / np.sqrt(data / n_shots)

err_diff = [{'err': mean_diff / snr(sum_diff, prof_sum, n),
                'n_shots': n} for n in config['n_shots']]

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
                              label=f"Error for {err_diff[count]['n_shots']} shots")
        count += 1
    fill = (mean_diff + err_diff[-1]['err'], mean_diff - err_diff[-1]['err'])
    plt.fill_between(qrange, fill[0], fill[1], 
                          color='lightsteelblue', edgecolor='lightsteelblue', alpha=0.8,
                          label=f"Error for {err_diff[-1]['n_shots']} shots")

    plt.plot(qrange, mean_diff, '--', color='black', label='Difference Profile')
    plt.xlim(qrange[0], qrange[-1])
    plt.ylim(-np.max(mean_diff)*(1-0.9), np.max(mean_diff)*(1+0.35))
    plt.xlabel(r"q = 4 $\pi$ $\sin\theta$ / $\lambda$  $[\AA^{-1}]$", fontsize=14)
    plt.ylabel("Mean Difference Profile, I(q)", fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(alpha=0.3)
    plt.title(title_sem, fontsize=12)
    plt.show()












































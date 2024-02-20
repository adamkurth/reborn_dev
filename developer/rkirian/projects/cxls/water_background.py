import sys
import os
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
import h5py
import reborn
from reborn.simulate import solutions, form_factors
from reborn.viewers.qtviews import PADView
from reborn.detector import RadialProfiler
from scipy import constants as const

from reborn.external import crystfel 
# crystfel.py interested in load_crystfel_geometry and geometry_dict_to_pad_geometry_list

def save_h5(data, filepath):
    with h5py.File(filepath, 'w') as file:
        file.create_dataset('entry/data/data', data=data)
    print(f"Data saved to {filepath} in entry/data/data")

def apply_water_image(data, water_image):
    print(f"Applying water image of shape {water_image.shape} to data of shape {data.shape}")
    return data * water_image

r_e = const.value('classical electron radius')
eV = const.value('electron volt')

##########################################################
# Configuration (everything in SI units)
#############################################################
# detector_shape = [2167, 2167]
detector_shape = [2162, 2068] # for Eiger4M
pixel_size = 75e-6
detector_distance = 0.3  # Sample to detector distance
sample_thickness = 100e-6  # Assuming a sheet of liquid of this thickness
n_shots = 1000  # Number of shots to integrate
n_photons = 1e7  # Photons per shot
photon_energy = 8000*eV  # Photon energy
beam_divergence = 2e-3  # Beam divergence (assuming this limits small-q)
beam_diameter = 5e-6  # X-ray beam diameter (doesn't really matter for solutions scattering)
protein_radius = 10e-9  # Radius of our spherical protein
protein_density = 1.34 * 1e3  # Density of spherical protein (g/cm^3, convert to SI kg/m^3)
protein_concentration = 10  #  Concentration of protein (mg/ml, which is same as SI kg/m^3)

###############################################################################################
# The above parameters are configurable.  Don't add new config parameters below this point!
#########################################################################################
# pad = reborn.detector.PADGeometry(distance=detector_distance, shape=detector_shape, pixel_size=pixel_size)

# added by adam

# relative paths to grab geometry file
cwd = os.getcwd()
base_dir_index = cwd.find('reborn_dev') # or 'reborn'
if base_dir_index != -1:
    base_dir_path = cwd[:base_dir_index + len('reborn_dev')]
    water_background_path = os.path.join(base_dir_path, "developer/rkirian/projects/water/adam")
    geom_file_path = os.path.join(base_dir_path, "developer/rkirian/projects/water/adam/Eiger4M.geom")
else:
    print("Base directory 'reborn_dev' not found in the current working directory")

geom_dictonary = crystfel.load_crystfel_geometry(geom_file_path)
pad = crystfel.geometry_dict_to_pad_geometry_list(geom_dictonary)[0]
print(geom_dictonary)
print(pad)

# end of added by adam

beam = reborn.source.Beam(photon_energy=photon_energy, diameter_fwhm=beam_diameter, pulse_energy=n_photons*photon_energy)
mask = pad.beamstop_mask(beam=beam, min_angle=beam_divergence)
n_water_molecules = sample_thickness * np.pi * (beam.diameter_fwhm/2)**2 * solutions.water_number_density()
m_protein = protein_density * 4 * np.pi * protein_radius ** 3 / 3  # Spherical protein mass
n = protein_concentration / m_protein  # Number density of spherical proteins
n_protein_molecules = sample_thickness * np.pi * (beam.diameter_fwhm/2)**2 * n
q = pad.q_vecs(beam=beam)
q_mags = pad.q_mags(beam=beam)
J = beam.photon_number_fluence
P = pad.polarization_factors(beam=beam)
SA = pad.solid_angles()
F_water = solutions.get_water_profile(q_mags)
F2_water = F_water**2*n_water_molecules
F_sphere = form_factors.sphere_form_factor(radius=protein_radius, q_mags=q_mags)
F_sphere *= (protein_density - 1000)/1000 * 3.346e29  # Protein-water contrast.  Water electron density is 3.35e29.
F2_sphere = n_protein_molecules * np.abs((F_sphere**2))
F2 = F2_water + F2_sphere
I = n_shots * r_e**2 * J * P * SA * F2
I = np.random.poisson(I)

profiler = RadialProfiler(pad_geometry=pad, beam=beam, n_bins=500, q_range=(0, np.max(q_mags)))
prof = profiler.get_mean_profile(I)
I = pad.reshape(I)
I *= mask.astype(int)

# save/write hdf5 file
water_image_path = os.path.join(water_background_path, 'water_background.h5')
# save_h5(I, water_image_path) # save the water image





# 1. simulate images (not water background) with specific protein and first detector distance 
# 2. keep these images on agave, but apply the water background to these images and copy them to a new directory
#    - make sure there are empty images (only water in the image) and images with peaks
# 3. train the neural network on the images with the water background and normal images
# 4. repeat steps 1-3 for the second detector distance


# feed this through the simulated data in agave
# of the simulated images(1) add the water background image (2) to produce a new image
# these are the images that we will train the neural network on
# empty image (only water in the image), is bad 
# with peaks, is good
# train on two different detector distances .


if 'noplots' not in sys.argv:

    pg.plot(np.log10(prof + 0.0001))
    padview = PADView(pad_geometry=[pad], raw_data=[I])
    padview.start()

    # plt.imshow(I, cmap='gray', interpolation='nearest')
    # plt.colorbar()
    # plt.title('water. %g µm pix.  %g m dist. %g mJ' % (pad.pixel_size()*1e6, pad.t_vec.flat[2], beam.pulse_energy*1e3))
    # plt.show()

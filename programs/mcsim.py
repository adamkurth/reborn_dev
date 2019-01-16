#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import sys
import time
import numpy as np
import h5py
from glob import glob
from scipy.special import erf

sys.path.append("..")  # This path setting won't be needed once bornagain is properly installed
import bornagain as ba
from bornagain.simulate import solutions
from bornagain.simulate import simutils
from bornagain.utils import vec_mag
from bornagain.units import r_e, hc, keV
from bornagain import target
import bornagain.simulate.clcore as core


def mcsim(detector_distance=100e-3, pixel_size=110e-6, n_pixels=1000, \
          beam_diameter=10e-6, photon_energy=12.0, n_photons=1e8, \
          mosaicity_fwhm=1e-4, beam_divergence_fwhm=1e-2, photon_energy_fwhm=0.02, \
          beam_spatial_profile='tophat', crystal_size=10e-6, crystal_size_fwhm=0.0, \
          mosaic_domain_size=1e-6, mosaic_domain_size_fwhm=0.0, \
          water_radius=0.0, temperature=298.16, \
          n_monte_carlo_iterations=1000, num_patterns=1, seed=0, random_rotation=True, \
          approximate_shape_transform=True, cromer_mann=False, expand_symm=False, \
          fix_rot_seq=False, mask_direct_beam=True, \
          pdb_file='../examples/data/pdb/2LYZ-P1.pdb', \
          write_hdf5=True, write_geom=True, write_crystal_sizes=True, \
          write_ideal_only=False, results_dir='/data/temp/', \
          quiet=False, compression=None, cl_double_precision=False):
    """
    This is a program to compute x-ray diffraction patterns using crystals.
	It can handle water scattering and uses a GPU to compute the diffraction pattern.
	The program can write geometry files, hdf5 files, and a file for the crystal sizes per pattern.
	Poisson noise is calculated per pattern in /data/noisy.
	All units are SI unless otherwise noted.
	
	Arguments:
		detector_distance (float) :
			Distance of detector from sample.
		pixel_size (float) :
			Side length of a pixel on the detector.
		n_pixels (int) :
			Number of pixels along side of detector.
		beam_diameter (float) :
			Diameter of direct beam.
		photon_energy (float) :
			Energy of incoming photons in keV.
		n_photons : 
			Number of photons in direct beam.
		mosaicity_fwhm (float) :
			FWHM of crystal mosaicity.
		beam_divergence_fwhm (float) :
			FWHM of direct beam divergence angle.
		photon_energy_fwhm (float) :
			FWHM of photon energies.
		beam_spatial_profile (string) :
			Spatial profile of intensities in direct beam. Options are 'tophat' and 'gaussian'.
		crystal_size (float) :
			Average side length of crystals.
		crystal_size_fwhm (float) :
			FWHM of crystal side lengths.
		mosaic_domain_size (float) :
			Average side length of mosaic domains.
		mosaic_domain_size_fwhm (float) : 
			FWHM of mosaic domain side lengths.
		water_radius (float) :
			Radius of water jet delivering crystals.
		temperature (float) :
			Temperature of water jet in Kelvin.
		n_monte_carlo_iterations (int) :
			Number of Monte Carlo iterations (>1000 recommended).
		num_patterns (int) :
			Number of patterns to simulate.
		seed (int) : 
			If the seed is fixed, this is the seed.
		random_rotation (bool) :
			Whether to do random rotations of the crystal per pattern.
		approximate_shape_transform (bool) :
			Whether to use a gaussian shape transform or a parallelepiped.
		cromer_mann (bool) : 
			Whether to use cromer_mann for molecular transform.
		expand_symmetry (bool) : 
			Whether to expand crystal symmetry.
		fix_rot_seq (bool) :
			Fixes the rotation sequence and seed.
		mask_direct_beam (bool) : 
			Whether to mask the direct beam on the patterns.
		pdb_file (string) :
			Path to PDB file to use for generating a crystal.
		write_hdf5 (bool) : 
			Whether to write intensities to an HDF5 file.
		write_geom (bool) :
			Whether to write detector geometry file.
		write_crystal_size (bool) :
			Whether to write crystal sizes per pattern to a text file.
		write_ideal_only (bool) : 
			Whether to write only the ideal pattern or include the noisy pattern.
		results_dir (string) : 
			Path to directory to store results.
		quiet (bool) : 
			Whether to suppress text output to screen.
		compression (bool) : 
			Whether to compress HDF5 files.
		cl_double_precision (bool) :
			Whether to use double-precision for GPU calculations.

	Returns:
		None
    """

# FIXME: check that the hdf5 file conforms to cxidb format.


    # Beam parameters
    photon_energy                   = photon_energy / keV
    wavelength                      = hc / photon_energy # pulse_energy = 0.0024
    wavelength_fwhm                 = wavelength * photon_energy_fwhm
    n_photons                       = int(n_photons) # pulse_energy / photon energy
    I0                              = n_photons / (beam_diameter ** 2) # Square beam

    # Crystal parameters
    crystal_size_fwhm               = crystal_size * crystal_size_fwhm
    mosaic_domain_size_fwhm         = mosaic_domain_size * mosaic_domain_size_fwhm

    # Rotation parameters
    rotation_axis                   = [1, 0, 0]
    rotation_angle                  = 0.1

    # Misc. parameters
    n_pixels                        = int(n_pixels)
    n_monte_carlo_iterations        = int(n_monte_carlo_iterations)
    num_patterns                    = int(num_patterns)

    # Handle argument errors before computing
    if temperature<0 or mosaicity_fwhm<0 or photon_energy_fwhm<0 or \
        beam_divergence_fwhm<0 or crystal_size_fwhm<0 or \
        mosaic_domain_size_fwhm<0 or water_radius<0:
        sys.exit('ERROR: one or more of you parameters is an invalid negative value')

    if n_monte_carlo_iterations < 1:
        sys.exit('ERROR: iterations must be an integer larger than zero')

    if beam_diameter<=0:
        sys.exit('ERROR: beam diameter must be greater than zero')

    if detector_distance<=0:
        sys.exit('ERROR: detector distance must be greater than zero')

    if pixel_size<=0:
        sys.exit('ERROR: pixel_size must be greater than zero')

    if n_pixels<=0:
        sys.exit('ERROR: n_pixels must be greater than zero')

    if mosaic_domain_size<=0:
        sys.exit('ERROR: mosaic_domain_size must be greater than zero')

    if crystal_size<=0:
        sys.exit('ERROR: crystal size must be greater than zero')

    if photon_energy<=0:
        sys.exit('ERROR: photon_energy must be greater than zero')

    if num_patterns<=0:
        sys.exit('ERROR: num_patterns must be greater than zero')

    if n_photons<=0:
        sys.exit('ERROR: n_photons must be greater than zero')

    if beam_spatial_profile != 'tophat' and beam_spatial_profile != 'gaussian':
        sys.exit('ERROR: beam_spatial_profile must be either gaussian or tophat')

    if not os.path.isfile(pdb_file):
        sys.exit('ERROR: pdb file does not exist')

    if not (compression == None or compression == 'lzf' or compression == 'gzip'):
        sys.exit('ERROR: compression format must be either lzf or gzip')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Creating text file with output parameters
    values = [detector_distance, pixel_size, n_pixels, beam_diameter, photon_energy * keV, n_photons, mosaicity_fwhm,
              beam_divergence_fwhm, photon_energy_fwhm, beam_spatial_profile, crystal_size, crystal_size_fwhm / crystal_size, mosaic_domain_size,
              mosaic_domain_size_fwhm / mosaic_domain_size, water_radius, temperature, n_monte_carlo_iterations, num_patterns, seed,
              random_rotation, approximate_shape_transform, cromer_mann, expand_symm, fix_rot_seq, mask_direct_beam,
              pdb_file, write_hdf5, write_geom, write_crystal_sizes, write_ideal_only, results_dir, quiet, compression, cl_double_precision]

    names  = ['detector_distance', 'pixel_size', 'n_pixels', 'beam_diameter', 'photon_energy', 'n_photons', 
              'mosaicity_fwhm', 'beam_divergence_fwhm', 'photon_energy_fwhm', 'beam_spatial_profile', 'crystal_size', 'crystal_size_fwhm',
              'mosaic_domain_size', 'mosaic_domain_size_fwhm', 'water_radius', 'temperature', 'n_monte_carlo_iterations', 'num_patterns',
              'seed', 'random_rotation', 'approximate_shape_transform', 'cromer_mann', 'expand_symm', 'fix_rot_sequence', 'mask_direct_beam',
              'pdb_file', 'write_hdf5', 'write_geom', 'write_crystal_sizes', 'write_ideal_only', 'results_dir', 'quiet', 'compression', 'cl_double_precision']


    pseudo_dict = zip(names, values)
    dictionary  = dict(pseudo_dict)
# FIXME: Just create the above dictionary directly; why make the two lists first?
    file_name = os.path.join(results_dir, 'used_params.txt')
    used_params = open(str(file_name), 'w+')

    for k, v in dictionary.items():
        if v != str(v):
            v = str(v)
        used_params.write(k + '=' + v + '\n')

    if not quiet:
        write = sys.stdout.write
    else:
        write = lambda x: x

    section = '=' * 70 + '\n'

    write(section)
    if fix_rot_seq:
        write("Fixing the rotation sequence based on random seed")
    write('PDB file: %s\n' % (os.path.basename(pdb_file)))
    write('Photons per pulse: %g\n' % (n_photons))
    write('Photon energy: %g keV\n' % (photon_energy * keV))
    write('Beam divergence: %g mrad FWHM\n' % (beam_divergence_fwhm * 1e3))
    write('Beam diameter: %g microns tophat\n' % (beam_diameter * 1e6))
    write('Spectral width: %g%% FWHM dlambda/lambda\n' % (100 * wavelength_fwhm / wavelength))
    write('Crystal size: %g microns\n' % (crystal_size * 1e6))
    write('Crystal mosaicity: %g radian FWHM\n' % (mosaicity_fwhm))
    write('Crystal mosaic domain size: %g microns\n' % (mosaic_domain_size * 1e6))
    write(section)

    # Things we probably don't want to think about
    cl_group_size = 32
    if(fix_rot_seq):
		np.random.seed(seed)

    # Setup simulation engine
    write('Setting up simulation engine... ')
    clcore = core.ClCore(group_size=cl_group_size, double_precision=cl_double_precision)
    write('done\n')

    write('Will run %d Monte Carlo iterations\n' % (n_monte_carlo_iterations))

    # Setup source info
    beam_vec = np.array([0,0,1])
    polarization_vec = np.array([1,0,0])
    polarization_weight = 1.0 # Fraction of polarization in this vector

    # Setup detector geometry
    write('Configuring detector... ')
    pad = ba.detector.PADGeometry()
    pad.simple_setup(n_pixels=n_pixels, pixel_size=pixel_size, distance=detector_distance)
    q = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
    qmag = vec_mag(q)
    sa = pad.solid_angles()
    P = pad.polarization_factors(polarization_vec=polarization_vec,beam_vec=beam_vec,weight=polarization_weight)
    write('done\n')

    # Get atomic coordinates and scattering factors from pdb file
    write('Getting atomic coordinates and scattering factors... ')
    if expand_symm:
        # TODO: eventually move the expand symmetry functionality to the crystal structure class
        cryst = target.crystal.Molecule(pdb_file)
        monomers = cryst.get_monomers()
        all_atoms = ba.target.crystal.Atoms.aggregate(monomers)
        r = all_atoms.xyz*1e-10
        Z = all_atoms.Z
    else:
        cryst = target.crystal.structure(pdb_file)
        r = cryst.r
        Z = cryst.Z

    f = ba.simulate.atoms.get_scattering_factors(Z, ba.units.hc / wavelength)
    write('done\n')
    write('%d atoms per unit cell\n' % (len(f)))

    # Do water scattering
    if water_radius!=0:
        write('Simulating water scattering... ')
        water_number_density = 33.3679e27
        illuminated_water_volume = simutils.volume_solvent(beam_diameter, crystal_size, water_radius)
        n_water_molecules = illuminated_water_volume * water_number_density
        F_water = solutions.get_water_profile(qmag, temperature=temperature) # Get water scattering intensity radial profile
        F2_water = F_water**2 * n_water_molecules
        I_water = I0 * r_e**2 * P * sa * F2_water
        if(illuminated_water_volume <= 0):
            write('\nWarning: No solvent was illuminated, water scattering not performed.\n')
            I_water = 0
        else:
            write('done\n')

    # Determine number of unit cells in whole crystal and mosaic domains
    n_cells_whole_crystal = np.ceil(crystal_size / np.array([cryst.a, cryst.b, cryst.c]))
    n_cells_mosaic_domain = np.ceil(mosaic_domain_size / np.array([cryst.a, cryst.b, cryst.c]))
    if(crystal_size > beam_diameter):
        n_cells_whole_crystal = np.ceil(np.array([beam_diameter, beam_diameter, crystal_size]) / np.array([cryst.a, cryst.b, cryst.c]))
    if(mosaic_domain_size > beam_diameter):
        n_cells_mosaic_domain = np.ceil(np.array([beam_diameter, beam_diameter, mosaic_domain_size]) / np.array([cryst.a, cryst.b, cryst.c]))


    # Setup function for shape transform calculations
    if approximate_shape_transform:
        write('Using approximate (Gaussian) shape transform\n')
        shape_transform = clcore.gaussian_lattice_transform_intensities_pad
    else:
        write('Using idealized (parallelepiped) shape transform\n')
        shape_transform = clcore.lattice_transform_intensities_pad

    if write_geom:
        geom_file = os.path.join(results_dir, 'geom.geom')
        write('Writing geometry file %s\n' % geom_file)
        fid = open(geom_file, 'w')
        fid.write("photon_energy = %g\n" % (photon_energy * ba.units.eV))
        fid.write("clen = %g\n" % detector_distance)
        fid.write("res = %g\n" % (1 / pixel_size))
        fid.write("adu_per_eV = %g\n" % (1.0 / (photon_energy * ba.units.eV)))
        fid.write("0/min_ss = 0\n")
        fid.write("0/max_ss = %d\n" % (n_pixels - 1))
        fid.write("0/min_fs = 0\n")
        fid.write("0/max_fs = %d\n" % (n_pixels - 1))
        fid.write("0/corner_x = %g\n" % (-n_pixels / 2.0))
        fid.write("0/corner_y = %g\n" % (-n_pixels / 2.0))
        fid.write("0/fs = x\n")
        fid.write("0/ss = y\n")
        fid.close()

    # Allocate memory on GPU device
    write('Allocating GPU device memory... ')
    r_dev = clcore.to_device(r, dtype=clcore.real_t)
    f_dev = clcore.to_device(f, dtype=clcore.complex_t)
    F_dev = clcore.to_device(np.zeros([int(pad.n_ss * pad.n_fs)], dtype=clcore.complex_t))
    S2_dev = clcore.to_device(shape=(int(pad.n_fs), int(pad.n_ss)), dtype=clcore.real_t)
    write('done\n')

    crystal_size_original = crystal_size
    mosaic_domain_size_original = mosaic_domain_size

    # Write text file containing crystal and mosaic domain sizes
    if write_crystal_sizes:
        file_name = os.path.join( results_dir, 'crystal_sizes' )
        cryst_size_file = open(file_name, 'w+')
        cryst_size_file.write('Crystal size (meters) : Mosaic domain size (meters) : Pattern file\n')

    for i in np.arange(1, (num_patterns + 1)):
        if(mosaic_domain_size_fwhm != 0):
            mosaic_domain_size = np.random.normal(mosaic_domain_size_original, mosaic_domain_size_fwhm / 2.354820045)
        if(crystal_size_fwhm != 0):
            crystal_size = np.random.normal(crystal_size_original, crystal_size_fwhm / 2.354820045)

            # Doesn't make sense if mosaic domain size is larger than the whole crystal...
            if mosaic_domain_size > crystal_size: mosaic_domain_size = crystal_size

            # Determine number of unit cells in whole crystal and mosaic domains
            n_cells_whole_crystal = np.ceil(crystal_size / np.array([cryst.a, cryst.b, cryst.c]))
            n_cells_mosaic_domain = np.ceil(mosaic_domain_size / np.array([cryst.a, cryst.b, cryst.c]))
            if(crystal_size > beam_diameter):
                n_cells_whole_crystal = np.ceil(np.array([beam_diameter, beam_diameter, crystal_size]) / np.array([cryst.a, cryst.b, cryst.c]))
            if(mosaic_domain_size > beam_diameter):
                n_cells_mosaic_domain = np.ceil(np.array([beam_diameter, beam_diameter, mosaic_domain_size]) / np.array([cryst.a, cryst.b, cryst.c]))

        # In case mosaic domain varied to be larger than fixed size crystal
        if mosaic_domain_size > crystal_size:
            mosaic_domain_size = crystal_size

        R = ba.utils.rotation_about_axis(rotation_angle, rotation_axis)

        if random_rotation:
            R = ba.utils.random_rotation()
        if not cromer_mann:
            write('Simulating molecular transform from Henke tables... ')
            t = time.time()
            clcore.phase_factor_pad(r_dev, f_dev, pad.t_vec, pad.fs_vec,
                pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R, F_dev, add=False)
            F2 = np.abs(F_dev.get()) ** 2
            tf = time.time() - t
            write('%g s\n' % (tf))
        else:
            write('Simulating molecular transform with cromer mann... ')
            t = time.time()
            clcore.prime_cromermann_simulator(q.copy(), Z.copy())
            q_cm = clcore.get_q_cromermann()
            r_cm = clcore.get_r_cromermann(r.copy(), sub_com=False)
            clcore.run_cromermann(q_cm, r_cm, rand_rot=False, force_rot_mat=R)
            A = clcore.release_amplitudes(reset=True)
            F2 = np.abs(A) ** 2
            tf = time.time() - t
            write('%g s\n' % (tf))

        abc = cryst.O.T.copy()
        S2_dev *= 0

        write('Simulating shape transform... ')
        time.sleep(0.001)
        message = ''
        tt = time.time()
        for n in np.arange(1, (n_monte_carlo_iterations + 1)):

            t = time.time()
            if (wavelength_fwhm > 0 or mosaicity_fwhm > 0 or beam_divergence_fwhm > 0):
                B = ba.utils.random_beam_vector(beam_divergence_fwhm)
                if (wavelength_fwhm == 0):
                    w = wavelength
                else:
                    w = np.random.normal(wavelength, wavelength_fwhm / 2.354820045, [1])[0]
                Rm = ba.utils.random_mosaic_rotation(mosaicity_fwhm).dot(R)
                T = pad.t_vec.copy() + pad.fs_vec * (np.random.random([1]) - 0.5) + pad.ss_vec * (np.random.random([1]) - 0.5)
            else:
                B = beam_vec
                w = wavelength
                Rm = R
                T = pad.t_vec

            shape_transform(abc, n_cells_mosaic_domain, T, pad.fs_vec, pad.ss_vec, B, pad.n_fs, pad.n_ss, w, Rm, S2_dev, add=True)

            tf = time.time() - t
            if (n % 1000) == 0:
                write('\b' * len(message))
                message = '%3.0f%% (%5d; %7.03f ms)' % (n / float(n_monte_carlo_iterations) * 100, n, tf * 1e3)
                write(message)
        write('\b' * len(message))
        write('%g s                \n' % (time.time() - tt))
        # Average the shape transforms over MC iterations
        S2 = S2_dev.get().ravel() / n
        # Convert into useful photon units
        I = I0 * r_e ** 2 * F2 * S2 *  sa * P
        if(crystal_size < beam_diameter): # Correct for lower incident intensity
            if(beam_spatial_profile == 'gaussian'):
                sig = beam_diameter / 3.0 # Let beam_diameter be 3 sigmas
                I *= erf(crystal_size/(sig * np.sqrt(2)))
            else:
                I *= (crystal_size/beam_diameter)**2

		# Make a mask for the direct beam
        if(mask_direct_beam):
            direct_beam_mask = np.ones((int(pad.n_ss * pad.n_fs)))
            direct_beam_mask[qmag < (2 * np.pi / np.max(np.array([cryst.a, cryst.b, cryst.c])))] = 0
            I *= direct_beam_mask

        # Scale up according to mosaic domain
        n_domains = np.prod(n_cells_whole_crystal) / np.prod(n_cells_mosaic_domain)
        I_ideal = I.copy() * n_domains

        if(water_radius != 0):
            I_ideal += I_water
        I_ideal = I_ideal.astype(np.float32)
        I_noisy = np.random.poisson(I_ideal).astype(np.float32)

        if write_hdf5:
            n_patterns = len(glob( os.path.join(results_dir , 'pattern-*.h5')))
            file_name = os.path.join( results_dir , 'pattern-%06d.h5' % (n_patterns + 1))
            write('Writing file %s\n' % file_name)
            fid = h5py.File(file_name, 'w')
            fid['/data/ideal'] = I_ideal.astype(np.float32).reshape((pad.n_ss, pad.n_fs))
            fid['/data/noisy'] = I_noisy.astype(np.int32).reshape((pad.n_ss, pad.n_fs))
            if(water_radius > 0):
                fid['/data/water'] = I_water.astype(np.float32).reshape((pad.n_ss, pad.n_fs))
            fid.close()

            with h5py.File( file_name,  'w') as fid:
                sh = (int(pad.n_ss), int(pad.n_fs))
                fid.create_dataset("data/ideal",
                    data= I_ideal.astype(np.float32).reshape(sh),
                    compression=compression, shape=sh)
                if not write_ideal_only:
                    fid.create_dataset("data/noisy",
                        data= I_noisy.astype(np.float32).reshape(sh),
                        compression=compression, shape=sh)
                if water_radius > 0:
                    fid.create_dataset('data/water',
                        data=I_water.astype(np.float32).reshape(sh),
                        compression=compression, shape=sh)
                fid.create_dataset("rotation_matrix", data=R)

        if write_crystal_sizes:
            cryst_size_file.write('%g:%g:pattern-%06d.h5\n' % (crystal_size, mosaic_domain_size, (n_patterns + 1)))

        # F2mean = np.mean(F2.flat[direct_beam_mask > 0])
        # Ncells = np.prod(n_cells_whole_crystal)
        # darwin = I0 * r_e ** 2 * Ncells * F2mean * (wavelength ** 3 / cryst.V)

    # End of pattern loop
    if write_crystal_sizes:
        cryst_size_file.close()

    write("\n\nDone!\n\n")

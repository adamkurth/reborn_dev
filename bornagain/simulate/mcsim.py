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
import bornagain.simulate.clcore as core

def mcsim(detector_distance=100e-3, pixel_size=110e-6, n_pixels=1000, \
          beam_diameter=10e-6, photon_energy=12.0, n_photons=1e8, \
          mosaicity_fwhm=1e-4, beam_divergence_fwhm=1e-2, beam_spatial_profile='tophat', \
          photon_energy_fwhm=0.02, crystal_size=10e-6, crystal_size_fwhm=0.0, \
          mosaic_domain_size=1e-6, mosaic_domain_size_fwhm=0.0, \
          n_monte_carlo_iterations=1000, num_patterns=1, random_rotation=True, \
          approximate_shape_transform=True, cromer_mann=False, expand_symm=False, \
          fix_rot_seq=False, mask_direct_beam=False, \
          pdb_file='../../examples/data/pdb/2LYZ-P1.pdb', \
          write_ideal_only=False, \
          quiet=False, cl_double_precision=False):
    """
    TODO: Write docstring.
    """

# FIXME: In general, the code within the bornagain/bornagain directory should consist of a library of classes, functions, etc. that we use to write programs.  AS it stands, this code is more like a program than a library.  We need to separate the core simulator from all the print statements, file reading/writing, etc.  The mcsim program should go into the programs directory.

# FIXME: we want "mcsim core"  - should be a class, which holds clcore instance.
    # Input : r (atomic pos.), f (structure factors), PADGeometry, Beam class instance.
    # Should not do any writing to terminal.
    # class called McCore
    # Start with does not do crystals
    # Output: intensities
    # Eventually: allow for repeats of the molecule.  one case is a set of translations.  Special case: crystals.

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
    if  mosaicity_fwhm<0 or photon_energy_fwhm<0 or \
        beam_divergence_fwhm<0 or crystal_size_fwhm<0 or \
        mosaic_domain_size_fwhm<0:
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

    # Things we probably don't want to think about
    cl_group_size = 32

    # Setup simulation engine
    clcore = core.ClCore(group_size=cl_group_size, double_precision=cl_double_precision)

    # Setup source info
    beam_vec = np.array([0,0,1])
    polarization_vec = np.array([1,0,0])
    polarization_weight = 1.0 # Fraction of polarization in this vector

    # Setup detector geometry
    pad = ba.detector.PADGeometry()
    pad.simple_setup(n_pixels=n_pixels, pixel_size=pixel_size, distance=detector_distance)
    q = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
    qmag = vec_mag(q)
    sa = pad.solid_angles()
    P = pad.polarization_factors(polarization_vec=polarization_vec,beam_vec=beam_vec,weight=polarization_weight)

    # Get atomic coordinates and scattering factors from pdb file
    if expand_symm:
        # TODO: eventually move the expand symmetry functionality to the crystal structure class
        cryst = ba.target.crystal.Molecule(pdb_file)
        monomers = cryst.get_monomers()
        all_atoms = ba.target.crystal.Atoms.aggregate(monomers)
        r = all_atoms.xyz*1e-10
        Z = all_atoms.Z
    else:
        cryst = ba.target.crystal.structure(pdb_file)
        r = cryst.r
        Z = cryst.Z

    f = ba.simulate.atoms.get_scattering_factors(Z, ba.units.hc / wavelength)

    # Determine number of unit cells in whole crystal and mosaic domains
    n_cells_whole_crystal = np.ceil(crystal_size / np.array([cryst.a, cryst.b, cryst.c]))
    n_cells_mosaic_domain = np.ceil(mosaic_domain_size / np.array([cryst.a, cryst.b, cryst.c]))
    if(crystal_size > beam_diameter):
        n_cells_whole_crystal = np.ceil(np.array([beam_diameter, beam_diameter, crystal_size]) / np.array([cryst.a, cryst.b, cryst.c]))
    if(mosaic_domain_size > beam_diameter):
        n_cells_mosaic_domain = np.ceil(np.array([beam_diameter, beam_diameter, mosaic_domain_size]) / np.array([cryst.a, cryst.b, cryst.c]))

    # Make a mask for the direct beam
    direct_beam_mask = np.ones((int(pad.n_ss * pad.n_fs)))
    direct_beam_mask[qmag < (2 * np.pi / np.max(np.array([cryst.a, cryst.b, cryst.c])))] = 0

    # Setup function for shape transform calculations
    if approximate_shape_transform:
        shape_transform = clcore.gaussian_lattice_transform_intensities_pad
    else:
        shape_transform = clcore.lattice_transform_intensities_pad

    # Allocate memory on GPU device
    r_dev = clcore.to_device(r, dtype=clcore.real_t)
    f_dev = clcore.to_device(f, dtype=clcore.complex_t)
    F_dev = clcore.to_device(np.zeros([int(pad.n_ss * pad.n_fs)], dtype=clcore.complex_t))
    S2_dev = clcore.to_device(shape=(int(pad.n_fs), int(pad.n_ss)), dtype=clcore.real_t)

    crystal_size_original = crystal_size
    mosaic_domain_size_original = mosaic_domain_size

    for i in np.arange(1, (num_patterns + 1)):
        # FIXME: the stuff within this loop should probably be made into a separate class, along with the ClCore() instance at the top.
        #        that way we can do monte-carlo simulations within various programs, not just for crystallography.
        if(mosaic_domain_size_fwhm != 0):
            mosaic_domain_size = np.random.normal(mosaic_domain_size_original, mosaic_domain_size_fwhm / 2.354820045)
        if(crystal_size_fwhm != 0):
            crystal_size = np.random.normal(crystal_size_original, crystal_size_fwhm / 2.354820045)

            # Doesn't make sense if mosaic domain size is larger than the whole crystal...
            if mosaic_domain_size > crystal_size: mosaic_domain_size = crystal_size

            # Determine number of illuminated unit cells in whole crystal and mosaic domains
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
            if fix_rot_seq:
# FIXME: good to set the seed on request, but I think this should be done at the very top so it affects all random numbers
                np.random.seed(i)
            R = ba.utils.random_rotation()
        if not cromer_mann:
            clcore.phase_factor_pad(r_dev, f_dev, pad.t_vec, pad.fs_vec,
                pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss, wavelength, R, F_dev, add=False)
            F2 = np.abs(F_dev.get()) ** 2
        else:
            clcore.prime_cromermann_simulator(q.copy(), Z.copy())
            q_cm = clcore.get_q_cromermann()
            r_cm = clcore.get_r_cromermann(r.copy(), sub_com=False)
            clcore.run_cromermann(q_cm, r_cm, rand_rot=False, force_rot_mat=R)
            A = clcore.release_amplitudes(reset=True)
            F2 = np.abs(A) ** 2

        abc = cryst.O.T.copy()
        S2_dev *= 0

        time.sleep(0.001)
        for n in np.arange(1, (n_monte_carlo_iterations + 1)):

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

        # Scale up according to mosaic domain
        n_domains = np.prod(n_cells_whole_crystal) / np.prod(n_cells_mosaic_domain)
        I_ideal = I.copy() * n_domains

        I_ideal = I_ideal.astype(np.float32)
        I_noisy = np.random.poisson(I_ideal).astype(np.float32)

        # F2mean = np.mean(F2.flat[direct_beam_mask > 0])
        # Ncells = np.prod(n_cells_whole_crystal)
        # darwin = I0 * r_e ** 2 * Ncells * F2mean * (wavelength ** 3 / cryst.V)

    # End of pattern loop

mcsim(detector_distance=100e-3, pixel_size=110e-6, n_pixels=1000, \
          beam_diameter=10e-6, photon_energy=12.0, n_photons=1e9, \
          mosaicity_fwhm=1e-4, beam_divergence_fwhm=1e-2, beam_spatial_profile='tophat', \
          photon_energy_fwhm=0.02, crystal_size=10e-6, crystal_size_fwhm=0.0, \
          mosaic_domain_size=1e-6, mosaic_domain_size_fwhm=0.0, \
          n_monte_carlo_iterations=1000, num_patterns=1, random_rotation=True, \
          approximate_shape_transform=True, cromer_mann=False, expand_symm=False, \
          fix_rot_seq=False, mask_direct_beam=False, \
          pdb_file='../../examples/data/pdb/2LYZ-P1.pdb', \
          write_ideal_only=False, \
          quiet=False, cl_double_precision=False)

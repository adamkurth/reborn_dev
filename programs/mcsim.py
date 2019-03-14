#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import h5py
from glob import glob
from scipy.special import erf
import bornagain as ba
from bornagain.simulate import solutions
from bornagain.simulate import simutils
from bornagain.utils import vec_mag
from bornagain.units import r_e, hc, keV
import bornagain.simulate.clcore as core

# This path setting won't be needed once bornagain is properly installed
sys.path.append("..")


def mcsim(
        detector_distance=50e-3,
        pixel_size=110e-6,
        n_pixels=1000,
        beam_diameter=1e-6,
        photon_energy=9.0,
        n_photons=1e12,
        transmission=1.0,
        mosaicity_fwhm=0e-4,
        beam_divergence_fwhm=0e-2,
        beam_spatial_profile='tophat',
        photon_energy_fwhm=0.0,
        crystal_size=1e-6,
        crystal_size_fwhm=0e-6,
        mosaic_domain_size=0.1e-6,
        mosaic_domain_size_fwhm=0.0,
        water_radius=0.0,
        temperature=298.16,
        n_monte_carlo_iterations=1,
        num_patterns=1,
        random_rotation=True,
        approximate_shape_transform=True,
        cromer_mann=False,
        expand_symm=False,
        fix_rot_seq=False,
        overlay_wigner_cells=False,
        mask_direct_beam=False,
        pdb_file='../../examples/data/pdb/2LYZ-P1.pdb',
        write_hdf5=True,
        write_geom=True,
        write_crystal_sizes=True,
        write_ideal_only=True,
        results_dir='./temp',
        quiet=False,
        compression=None,
        cl_double_precision=False):
    """
    TODO: Write docstring.
    """

    # Beam parameters
    photon_energy = photon_energy / keV
    wavelength = hc / photon_energy  # pulse_energy = 0.0024
    wavelength_fwhm = wavelength * photon_energy_fwhm # TODO: This is wrong: wavelength FWHM is not energy FWHM
    n_photons = int(n_photons)  # pulse_energy / photon energy
    I0 = transmission * n_photons / (beam_diameter ** 2)  # Square beam

    # Crystal parameters
    crystal_size_fwhm = crystal_size * crystal_size_fwhm
    mosaic_domain_size_fwhm = mosaic_domain_size * mosaic_domain_size_fwhm

    # Rotation parameters
    rotation_axis = [1, 0, 0]
    rotation_angle = 0.1

    # Misc. parameters
    n_pixels = int(n_pixels)
    n_monte_carlo_iterations = int(n_monte_carlo_iterations)
    num_patterns = int(num_patterns)

    # Handle argument errors before computing
    if temperature < 0 or mosaicity_fwhm < 0 or photon_energy_fwhm < 0 or \
            beam_divergence_fwhm < 0 or crystal_size_fwhm < 0 or \
            mosaic_domain_size_fwhm < 0 or water_radius < 0:
        sys.exit('ERROR: one or more of you parameters is an invalid negative value')

    if n_monte_carlo_iterations < 1:
        sys.exit('ERROR: iterations must be an integer larger than zero')

    if beam_diameter <= 0:
        sys.exit('ERROR: beam diameter must be greater than zero')

    if detector_distance <= 0:
        sys.exit('ERROR: detector distance must be greater than zero')

    if pixel_size <= 0:
        sys.exit('ERROR: pixel_size must be greater than zero')

    if n_pixels <= 0:
        sys.exit('ERROR: n_pixels must be greater than zero')

    if mosaic_domain_size <= 0:
        sys.exit('ERROR: mosaic_domain_size must be greater than zero')

    if crystal_size <= 0:
        sys.exit('ERROR: crystal size must be greater than zero')

    if photon_energy <= 0:
        sys.exit('ERROR: photon_energy must be greater than zero')

    if num_patterns <= 0:
        sys.exit('ERROR: num_patterns must be greater than zero')

    if n_photons <= 0:
        sys.exit('ERROR: n_photons must be greater than zero')

    if transmission < 0.0 or transmission > 1.0:
        sys.exit('ERROR: transmission must be between 0 and 1')

    if beam_spatial_profile != 'tophat' and beam_spatial_profile != 'gaussian':
        sys.exit('ERROR: beam_spatial_profile must be either gaussian or tophat')

    if not os.path.isfile(pdb_file):
        sys.exit('ERROR: pdb file does not exist')

    if not (compression is None or compression ==
            'lzf' or compression == 'gzip'):
        sys.exit('ERROR: compression format must be either lzf or gzip')

    # Creating text file with output parameters
    values = [
        detector_distance,
        pixel_size,
        n_pixels,
        beam_diameter,
        photon_energy *
        keV,
        n_photons,
        transmission,
        mosaicity_fwhm,
        beam_divergence_fwhm,
        beam_spatial_profile,
        photon_energy_fwhm,
        crystal_size,
        crystal_size_fwhm /
        crystal_size,
        mosaic_domain_size,
        mosaic_domain_size_fwhm /
        mosaic_domain_size,
        water_radius,
        temperature,
        n_monte_carlo_iterations,
        num_patterns,
        random_rotation,
        approximate_shape_transform,
        cromer_mann,
        expand_symm,
        fix_rot_seq,
        overlay_wigner_cells,
        mask_direct_beam,
        pdb_file,
        write_hdf5,
        write_geom,
        write_crystal_sizes,
        write_ideal_only,
        results_dir,
        quiet,
        compression,
        cl_double_precision]

    names = [
        'detector_distance',
        'pixel_size',
        'n_pixels',
        'beam_diameter',
        'photon_energy',
        'n_photons',
        'transmission',
        'mosaicity_fwhm',
        'beam_divergence_fwhm',
        'beam_spatial_profile',
        'photon_energy_fwhm',
        'crystal_size',
        'crystal_size_fwhm',
        'mosaic_domain_size',
        'mosaic_domain_size_fwhm',
        'water_radius',
        'temperature',
        'n_monte_carlo_iterations',
        'num_patterns',
        'random_rotation',
        'approximate_shape_transform',
        'cromer_mann',
        'expand_symm',
        'fix_rot_sequence',
        'overlay_wigner_cells',
        'mask_direct_beam',
        'pdb_file',
        'write_hdf5',
        'write_geom',
        'write_crystal_sizes',
        'write_ideal_only',
        'results_dir',
        'quiet',
        'compression',
        'cl_double_precision']

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    pseudo_dict = zip(names, values)
    dictionary = dict(pseudo_dict)
    file_name = os.path.join(results_dir, 'used_params.txt')
    used_params = open(str(file_name), 'w+')

    for k, v in dictionary.items():
        if v != str(v):
            v = str(v)
        used_params.write(k + '=' + v + '\n')

    if not quiet:
        write = sys.stdout.write
    else:
        def write(x):
            return x

    section = '=' * 70 + '\n'

    write(section)
    if fix_rot_seq:
        write("Fixing the rotation sequence based on random seed")
    write('PDB file: %s\n' % (os.path.basename(pdb_file)))
    write('Photons per pulse: %g\n' % (n_photons))
    write('Photon energy: %g keV\n' % (photon_energy * keV))
    write('Beam divergence: %g mrad FWHM\n' % (beam_divergence_fwhm * 1e3))
    write('Beam diameter: %g microns tophat\n' % (beam_diameter * 1e6))
    write('Spectral width: %g%% FWHM dlambda/lambda\n' %
          (100 * wavelength_fwhm / wavelength))
    write('Crystal size: %g microns\n' % (crystal_size * 1e6))
    write('Crystal mosaicity: %g radian FWHM\n' % (mosaicity_fwhm))
    write(
        'Crystal mosaic domain size: %g microns\n' %
        (mosaic_domain_size * 1e6))
    write(section)

    # Things we probably don't want to think about
    cl_group_size = 32

    # Setup simulation engine
    write('Setting up simulation engine... ')
    clcore = core.ClCore(
        group_size=cl_group_size,
        double_precision=cl_double_precision)
    write('done\n')

    write('Will run %d Monte Carlo iterations\n' % (n_monte_carlo_iterations))

    # Setup source info
    beam_vec = np.array([0, 0, 1])
    polarization_vec = np.array([1, 0, 0])
    polarization_weight = 1.0  # Fraction of polarization in this vector

    # Setup detector geometry
    write('Configuring detector... ')
    pad = ba.detector.PADGeometry()
    pad.simple_setup(n_pixels=n_pixels, pixel_size=pixel_size, distance=detector_distance)
    q = pad.q_vecs(beam_vec=beam_vec, wavelength=wavelength)
    qmag = vec_mag(q)
    sa = pad.solid_angles()
    P = pad.polarization_factors(polarization_vec=polarization_vec, beam_vec=beam_vec, weight=polarization_weight)
    write('done\n')

    # Get atomic coordinates and scattering factors from pdb file
    write('Getting atomic coordinates and scattering factors... ')
    if expand_symm:
        cryst = ba.target.crystal.Molecule(pdb_file)
        monomers = cryst.get_monomers()
        all_atoms = ba.target.crystal.Atoms.aggregate(monomers)
        r = all_atoms.xyz * 1e-10
        Z = all_atoms.Z
    else:
        cryst = ba.target.crystal.Structure(pdb_file)
        r = cryst.r
        Z = cryst.Z

    f = ba.simulate.atoms.get_scattering_factors(Z, ba.units.hc / wavelength)
    write('done\n')
    write('%d atoms per unit cell\n' % (len(f)))

    # Determine number of unit cells in whole crystal and mosaic domains
    n_cells_whole_crystal = np.ceil(
        crystal_size / np.array([cryst.a, cryst.b, cryst.c]))
    n_cells_mosaic_domain = np.ceil(
        mosaic_domain_size / np.array([cryst.a, cryst.b, cryst.c]))
    if(crystal_size > beam_diameter):
        n_cells_whole_crystal = np.ceil(np.array(
            [beam_diameter, beam_diameter, crystal_size]) / np.array([cryst.a, cryst.b, cryst.c]))
    if(mosaic_domain_size > beam_diameter):
        n_cells_mosaic_domain = np.ceil(np.array(
            [beam_diameter, beam_diameter, mosaic_domain_size]) / np.array([cryst.a, cryst.b, cryst.c]))

    # Make a mask for the direct beam
    direct_beam_mask = np.ones((int(pad.n_ss * pad.n_fs)))
    direct_beam_mask[qmag < (
        2 * np.pi / np.max(np.array([cryst.a, cryst.b, cryst.c])))] = 0

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
    F_dev = clcore.to_device(
        np.zeros([int(pad.n_ss * pad.n_fs)], dtype=clcore.complex_t))
    S2_dev = clcore.to_device(
        shape=(int(pad.n_fs), int(pad.n_ss)), dtype=clcore.real_t)
    write('done\n')

    crystal_size_original = crystal_size
    mosaic_domain_size_original = mosaic_domain_size

    # Write text file containing crystal and mosaic domain sizes
    if write_crystal_sizes:
        file_name = os.path.join(results_dir, 'crystal_sizes')
        cryst_size_file = open(file_name, 'w+')
        cryst_size_file.write(
            'Crystal size (meters) : Mosaic domain size (meters) : Pattern file\n')

    for i in np.arange(1, (num_patterns + 1)):
        if(mosaic_domain_size_fwhm != 0):
            mosaic_domain_size = np.random.normal(
                mosaic_domain_size_original,
                mosaic_domain_size_fwhm / 2.354820045)
        if(crystal_size_fwhm != 0):
            crystal_size = np.random.normal(
                crystal_size_original, crystal_size_fwhm / 2.354820045)

            # Doesn't make sense if mosaic domain size is larger than the whole
            # crystal...
            if mosaic_domain_size > crystal_size:
                mosaic_domain_size = crystal_size

            # Determine number of unit cells in whole crystal and mosaic
            # domains
            n_cells_whole_crystal = np.ceil(
                crystal_size / np.array([cryst.a, cryst.b, cryst.c]))
            n_cells_mosaic_domain = np.ceil(
                mosaic_domain_size / np.array([cryst.a, cryst.b, cryst.c]))
            if(crystal_size > beam_diameter):
                n_cells_whole_crystal = np.ceil(np.array(
                    [beam_diameter, beam_diameter, crystal_size]) / np.array([cryst.a, cryst.b, cryst.c]))
            if(mosaic_domain_size > beam_diameter):
                n_cells_mosaic_domain = np.ceil(np.array(
                    [beam_diameter, beam_diameter, mosaic_domain_size]) / np.array([cryst.a, cryst.b, cryst.c]))

        # In case mosaic domain varied to be larger than fixed size crystal
        if mosaic_domain_size > crystal_size:
            mosaic_domain_size = crystal_size

        # Do water scattering
        if water_radius > 0:
            write('Simulating water scattering... ')
            water_number_density = 33.3679e27
            illuminated_water_volume = simutils.volume_solvent(
                beam_diameter, crystal_size, water_radius)
            n_water_molecules = illuminated_water_volume * water_number_density
            # Get water scattering intensity radial profile
            F_water = solutions.get_water_profile(
                qmag, temperature=temperature)
            F2_water = F_water**2 * n_water_molecules
            I_water = I0 * r_e**2 * P * sa * F2_water
            if(illuminated_water_volume <= 0):
                write(
                    '\nWarning: No solvent was illuminated, water scattering not performed.\n')
                I_water = 0
            else:
                write('done\n')

        R = ba.utils.rotation_about_axis(rotation_angle, rotation_axis)
        if random_rotation:
            R = ba.utils.random_rotation()

        if random_rotation:
            if fix_rot_seq:
                np.random.seed(i)
            R = ba.utils.random_rotation()
        if not cromer_mann:
            write('Simulating molecular transform from Henke tables... ')
            t = time.time()
            clcore.phase_factor_pad(r_dev, f_dev, pad.t_vec, pad.fs_vec, pad.ss_vec, beam_vec, pad.n_fs, pad.n_ss,
                wavelength, R, F_dev, add=False)
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
        # time.sleep(0.001)
        message = ''
        tt = time.time()
        for n in np.arange(1, (n_monte_carlo_iterations + 1)):

            t = time.time()
            if (wavelength_fwhm > 0 or mosaicity_fwhm >
                    0 or beam_divergence_fwhm > 0):
                B = ba.utils.random_beam_vector(beam_divergence_fwhm)
                if (wavelength_fwhm == 0):
                    w = wavelength
                else:
                    w = np.random.normal(
                        wavelength, wavelength_fwhm / 2.354820045, [1])[0]
                Rm = ba.utils.random_mosaic_rotation(mosaicity_fwhm).dot(R)
                T = pad.t_vec.copy() + pad.fs_vec * \
                    (np.random.random([1]) - 0.5) + pad.ss_vec * (np.random.random([1]) - 0.5)
            else:
                B = beam_vec
                w = wavelength
                Rm = R
                T = pad.t_vec

            shape_transform(abc, n_cells_mosaic_domain, T, pad.fs_vec, pad.ss_vec, B, pad.n_fs, pad.n_ss, w, Rm, S2_dev,
                            add=True)

            tf = time.time() - t
            if (n % 1000) == 0:
                write('\b' * len(message))
                message = '%3.0f%% (%5d; %7.03f ms)' % (
                    n / float(n_monte_carlo_iterations) * 100, n, tf * 1e3)
                write(message)
        write('\b' * len(message))
        write('%g s                \n' % (time.time() - tt))
        # Average the shape transforms over MC iterations
        S2 = S2_dev.get().ravel() / n
        # Convert into useful photon units
        I = I0 * r_e ** 2 * sa * P * F2 * S2
        if(crystal_size < beam_diameter):  # Correct for lower incident intensity
            if(beam_spatial_profile == 'gaussian'):
                sig = beam_diameter / 3.0  # Let beam_diameter be 3 sigmas
                I *= erf(crystal_size / (sig * np.sqrt(2)))
            else:
                I *= (crystal_size / beam_diameter)**2

        # Scale up according to mosaic domain
        n_domains = np.prod(n_cells_whole_crystal) / np.prod(n_cells_mosaic_domain)
        I_ideal = I.copy() * n_domains

        if(water_radius != 0):
            I_ideal += I_water
        I_ideal = I_ideal.astype(np.float32)
        I_noisy = np.random.poisson(I_ideal).astype(np.float32)

        if write_hdf5:
            n_patterns = len(glob(os.path.join(results_dir, 'pattern-*.h5')))
            file_name = os.path.join(results_dir, 'pattern-%06d.h5' % (n_patterns + 1))
            write('Writing file %s\n' % file_name)
            fid = h5py.File(file_name, 'w')
            fid['/data/ideal'] = I_ideal.astype(np.float32).reshape((pad.n_ss, pad.n_fs))
            fid['/data/noisy'] = I_noisy.astype(np.int32).reshape((pad.n_ss, pad.n_fs))
            if(water_radius > 0):
                fid['/data/water'] = I_water.astype(np.float32).reshape((pad.n_ss, pad.n_fs))
            fid.close()
            write('Wrote file %s' % (file_name,))

            # FIXME: what's going on here -- why is the file being written twice?
            # with h5py.File(file_name, 'w') as fid:
            #     sh = (int(pad.n_ss), int(pad.n_fs))
            #     fid.create_dataset("data/ideal",
            #                        data=I_ideal.astype(np.float32).reshape(sh),
            #                        compression=compression, shape=sh)
            #     if not write_ideal_only:
            #         fid.create_dataset(
            #             "data/noisy",
            #             data=I_noisy.astype(
            #                 np.float32).reshape(sh),
            #             compression=compression,
            #             shape=sh)
            #     if water_radius > 0:
            #         fid.create_dataset(
            #             'data/water',
            #             data=I_water.astype(
            #                 np.float32).reshape(sh),
            #             compression=compression,
            #             shape=sh)
            #     fid.create_dataset("rotation_matrix", data=R)


        if write_crystal_sizes:
            cryst_size_file.write('%g:%g:pattern-%06d.h5\n' % (crystal_size, mosaic_domain_size, (n_patterns + 1)))

        # F2mean = np.mean(F2.flat[direct_beam_mask > 0])
        # Ncells = np.prod(n_cells_whole_crystal)
        # darwin = I0 * r_e ** 2 * Ncells * F2mean * (wavelength ** 3 / cryst.V)

    # End of pattern loop
    if write_crystal_sizes:
        cryst_size_file.close()

    write("\n\nDone!\n\n")


if __name__ == '__main__':

    import numpy as np
    import h5py
    import pyqtgraph
    from bornagain.simulate.examples import lysozyme_pdb_file, psi_pdb_file
    from bornagain.viewers.qtviews.padviews import PADView
    from bornagain.external.crystfel import geometry_file_to_pad_geometry_list
    import shutil
    import os

    if os.path.isdir('./temp'):
        shutil.rmtree('./temp')
    mcsim(pdb_file=psi_pdb_file, n_monte_carlo_iterations=1000)
    f = h5py.File('temp/pattern-000001.h5', 'r')
    data = np.array(f['/data/ideal'])
    geom_file = './temp/geom.geom'
    pad_geometry = geometry_file_to_pad_geometry_list(geom_file)
    padview = PADView(pad_geometry=pad_geometry, raw_data=[data])
    padview.show()
    pyqtgraph.QtGui.QApplication.exec_()
    if os.path.isdir('./temp'):
        shutil.rmtree('./temp')

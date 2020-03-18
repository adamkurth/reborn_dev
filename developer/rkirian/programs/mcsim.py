#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import h5py
import pickle
from glob import glob
import inspect
from scipy.special import erf
from scipy import constants
from scipy.spatial.transform import Rotation
import bornagain as ba
from bornagain.simulate import solutions
from bornagain.simulate import simutils
import bornagain.simulate.clcore as core
from bornagain.external import crystfel


r_e = constants.value('classical electron radius')
keV = 1000*constants.value('electron volt')
hc = constants.h*constants.c

def mcsim(
        detector_distance=50e-3,
        pixel_size=110e-6,
        n_pixels=1000,
        beam_diameter=1e-6,
        photon_energy=9.0,
        n_photons=1e12,
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
        pdb_file='../../examples/data/pdb/2LYZ-P1.pdb',
        write_hdf5=True,
        write_geom=True,
        results_dir='./temp',
        quiet=False,
        compression=None,
        cl_double_precision=False):
    r"""
    TODO: Write docstring.
    """

    if not quiet:
        write = sys.stdout.write
    else:
        def write(*args, **kwargs):
            pass

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

    if beam_spatial_profile != 'tophat' and beam_spatial_profile != 'gaussian':
        sys.exit('ERROR: beam_spatial_profile must be either gaussian or tophat')

    if not os.path.isfile(pdb_file):
        sys.exit('ERROR: pdb file does not exist')

    if not (compression is None or compression == 'lzf' or compression == 'gzip'):
        sys.exit('ERROR: compression format must be either lzf or gzip')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Beam parameters
    photon_energy = photon_energy / keV
    beam = ba.source.Beam(photon_energy=photon_energy)
    beam.photon_energy_fwhm = photon_energy_fwhm
    beam.diameter_fwhm = beam_diameter
    beam.beam_divergence_fwhm = beam_divergence_fwhm
    beam.pulse_energy = n_photons * photon_energy

    # Rotation parameters
    rotation_axis = [1, 0, 0]
    rotation_angle = 0.1

    # Misc. parameters
    n_pixels = int(n_pixels)
    n_monte_carlo_iterations = int(n_monte_carlo_iterations)
    num_patterns = int(num_patterns)

    # Setup detector geometry
    pad = ba.detector.PADGeometry(shape=(n_pixels, n_pixels), pixel_size=pixel_size, distance=detector_distance)
    # q_vecs = pad.q_vecs(beam=beam)
    qmag = pad.q_mags(beam=beam)
    sa = pad.solid_angles()
    pol = pad.polarization_factors(beam=beam)

    if write_geom:
        geom_file = os.path.join(results_dir, 'geom.geom')
        write('Writing geometry file %s\n' % geom_file)
        crystfel.write_geom_file_single_pad(file_path=geom_file, beam=beam, pad_geometry=pad)

    # Create text file with output parameters
    used_params = open(os.path.join(results_dir, 'used_params.txt'), 'w+')
    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    for i in args:
        used_params.write("%s = %s\n" % (i, values[i]))

    section = '=' * 70 + '\n'

    write(section)
    if fix_rot_seq:
        write("Fixing the rotation sequence based on random seed")
    write('PDB file: %s\n' % (os.path.basename(pdb_file)))
    write('Photons per pulse: %g\n' % n_photons)
    write('Photon energy: %g keV\n' % (photon_energy * keV))
    write('Beam divergence: %g mrad FWHM\n' % (beam.beam_divergence_fwhm * 1e3))
    write('Beam diameter: %g microns tophat\n' % (beam_diameter * 1e6))
    write('Spectral width: %g%% FWHM dE/E\n' % (100 * beam.photon_energy_fwhm / beam.photon_energy))
    write('Crystal size: %g microns\n' % (crystal_size * 1e6))
    write('Crystal mosaicity: %g radian FWHM\n' % (mosaicity_fwhm))
    write('Crystal mosaic domain size: %g microns\n' % (mosaic_domain_size * 1e6))
    write(section)

    # Get atomic coordinates and scattering factors from pdb file
    write('Getting atomic coordinates and scattering factors... ')
    cryst = ba.target.crystal.CrystalStructure(pdb_file)
    # Crystal parameters
    crystal_size_fwhm = crystal_size * crystal_size_fwhm
    mosaic_domain_size_fwhm = mosaic_domain_size * mosaic_domain_size_fwhm
    cryst.crystal_size = crystal_size
    cryst.crystal_size_fwhm = crystal_size * crystal_size_fwhm
    cryst.mosaic_domain_size = mosaic_domain_size
    cryst.mosaic_domain_size_fwhm = mosaic_domain_size * mosaic_domain_size_fwhm
    if expand_symm:
        write('\nExpanding symmetry... ')
        r = cryst.get_symmetry_expanded_coordinates()
        Z = cryst.molecule.atomic_numbers
        Z = np.concatenate([Z]*cryst.spacegroup.n_molecules)
    else:
        r = cryst.molecule.coordinates
        Z = cryst.molecule.atomic_numbers
    f = ba.simulate.atoms.get_scattering_factors(Z, photon_energy=beam.photon_energy)
    write('done\n')
    write('%d atoms per unit cell\n' % (len(f)))

    # Setup simulation engine
    write('Setting up simulation engine... ')
    cl_group_size = 32
    clcore = core.ClCore(group_size=cl_group_size, double_precision=cl_double_precision)
    # Allocate memory on GPU device
    write('\nAllocating GPU device memory... ')
    r_dev = clcore.to_device(r, dtype=clcore.real_t)
    f_dev = clcore.to_device(f, dtype=clcore.complex_t)
    F_dev = clcore.to_device(np.zeros([int(pad.n_ss * pad.n_fs)], dtype=clcore.complex_t))
    S2_dev = clcore.to_device(shape=(int(pad.n_fs), int(pad.n_ss)), dtype=clcore.real_t)
    write('done\n')

    # Setup function for shape transform calculations
    if approximate_shape_transform:
        write('Using approximate (Gaussian) shape transform\n')
        shape_transform = clcore.gaussian_lattice_transform_intensities_pad
    else:
        write('Using idealized (parallelepiped) shape transform\n')
        shape_transform = clcore.lattice_transform_intensities_pad

    # Do water scattering
    if water_radius > 0:
        write('Simulating water scattering... ')
        illuminated_water_volume = simutils.volume_solvent(beam_diameter, crystal_size, water_radius)
        F2_water = solutions.get_water_profile(qmag, temperature=temperature, volume=illuminated_water_volume)
        I_water = I0 * r_e ** 2 * pol * sa * F2_water
        write('done.\n')

    write('Will run %d Monte Carlo iterations\n' % n_monte_carlo_iterations)

    beam_area = np.pi * beam_diameter ** 2 / 4.0
    cell_volume = cryst.unitcell.volume

    for i in np.arange(1, (num_patterns + 1)):

        if fix_rot_seq:
            np.random.seed(i)

        this_mosaic_domain_size = np.random.normal(mosaic_domain_size, mosaic_domain_size_fwhm / 2.354820045)
        this_crystal_size = np.random.normal(crystal_size, crystal_size_fwhm / 2.354820045)

        # Determine number of unit cells in whole crystal and mosaic domains
        n_cells_whole_crystal = np.ceil(min(beam_area, this_crystal_size**2)*this_crystal_size / cell_volume)
        n_cells_mosaic_domain = np.ceil(min(beam_area, this_mosaic_domain_size**2)*this_mosaic_domain_size / cell_volume)

        if random_rotation:
            R = Rotation.random().as_matrix()
        else:
            R = ba.utils.rotation_about_axis(rotation_angle, rotation_axis)

        if not cromer_mann:
            write('Simulating molecular transform from Henke tables... ')
            t = time.time()
            clcore.phase_factor_pad(r_dev, f_dev, pad.t_vec, pad.fs_vec, pad.ss_vec, beam.beam_vec, pad.n_fs, pad.n_ss,
                                    beam.wavelength, R, F_dev, add=False)
            F2 = np.abs(F_dev.get()) ** 2
            tf = time.time() - t
            write('%g s\n' % tf)
        else:
            raise ValueError('Cromer-Mann needs to be re-implemented')
            # write('Simulating molecular transform with cromer mann... ')
            # t = time.time()
            # clcore.prime_cromermann_simulator(q.copy(), Z.copy())
            # q_cm = clcore.get_q_cromermann()
            # r_cm = clcore.get_r_cromermann(r.copy(), sub_com=False)
            # clcore.run_cromermann(q_cm, r_cm, rand_rot=False, force_rot_mat=R)
            # A = clcore.release_amplitudes(reset=True)
            # F2 = np.abs(A) ** 2
            # tf = time.time() - t
            # write('%g s\n' % tf)

        abc = cryst.O.T.copy()
        S2_dev *= 0

        write('Simulating shape transform... ')
        message = ''
        tt = time.time()
        for n in np.arange(1, (n_monte_carlo_iterations + 1)):

            t = time.time()
            if beam.photon_energy_fwhm > 0 or mosaicity_fwhm > 0 or beam.beam_divergence_fwhm > 0:
                B = ba.utils.random_beam_vector(beam.beam_divergence_fwhm)
                if wavelength_fwhm == 0:
                    w = wavelength
                else:
                    w = np.random.normal(wavelength, wavelength_fwhm / 2.354820045, [1])[0]
                Rm = ba.utils.random_mosaic_rotation(mosaicity_fwhm).dot(R)
                T = pad.t_vec + pad.fs_vec * (np.random.random([1]) - 0.5) + pad.ss_vec * (np.random.random([1]) - 0.5)
            else:
                B = beam.beam_vec
                w = beam.wavelength
                Rm = R
                T = pad.t_vec

            shape_transform(abc, np.array([n_cells_mosaic_domain**(1/3.)]*3), T, pad.fs_vec, pad.ss_vec, B, pad.n_fs,
                            pad.n_ss, w, Rm, S2_dev, add=True)

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
        I = beam.photon_number_fluence * r_e ** 2 * sa * pol * F2 * S2
        if crystal_size < beam_diameter:  # Correct for lower incident intensity
            if beam_spatial_profile == 'gaussian':
                sig = beam_diameter / 3.0  # Let beam_diameter be 3 sigmas
                I *= erf(crystal_size / (sig * np.sqrt(2)))
            else:
                I *= (crystal_size / beam_diameter)**2

        # Scale up according to mosaic domain
        n_domains = np.prod(n_cells_whole_crystal) / np.prod(n_cells_mosaic_domain)
        I_ideal = I.copy() * n_domains

        if water_radius != 0:
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
            fid['/cyrstal_size'] = crystal_size
            fid['/mosaic_domain_size'] = mosaic_domain_size
            if(water_radius > 0):
                fid['/data/water'] = I_water.astype(np.float32).reshape((pad.n_ss, pad.n_fs))
            fid.close()
            write('Wrote file %s' % (file_name,))

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
    mcsim(pdb_file=lysozyme_pdb_file, random_rotation=False, n_monte_carlo_iterations=100, photon_energy=5,
          detector_distance=0.4, expand_symm=True)
    f = h5py.File('temp/pattern-000001.h5', 'r')
    data = np.array(f['/data/ideal'])
    geom_file = './temp/geom.geom'
    pad_geometry = geometry_file_to_pad_geometry_list(geom_file)
    padview = PADView(pad_geometry=pad_geometry, raw_data=[data])
    padview.show()
    pyqtgraph.QtGui.QApplication.exec_()
    # if os.path.isdir('./temp'):
    #     shutil.rmtree('./temp')

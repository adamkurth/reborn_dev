"""
Simulate Ag Behenate crystal diffraction patterns.

Date Created: 21 Nov 2021
Last Modified: 21 Nov 2021
Author: RAK, JC
"""
import numpy as np
from scipy.spatial.transform import Rotation
from reborn import source, detector, dataframe, const, utils
from reborn.target import crystal, atoms
from reborn.simulate import clcore
from reborn.fileio.getters import FrameGetter
eV = const.eV
r_e = const.r_e
deg = np.pi/180
# ------------------------------
photon_energy = 9500*eV     # eV
diameter_fwhm = 1e-6      # m
pulse_energy = 1e-3       # J
detector_distance = 0.5   # m
pdb_id = '1507774.pdb'    # Hacked AgB "pdb" file
crystal_size = 100e-9     # Overall crystal size (affects integrated Bragg peak intensity)
crystal_domain_size = min(crystal_size, 100e-9)  # Domain size of crystal (affects Bragg peak size)
show_patterns = 1
add_poisson_noise = False     # Turn Poisson noise on or off
# ------------------------------
beam = source.Beam(photon_energy=photon_energy, diameter_fwhm=diameter_fwhm, pulse_energy=pulse_energy)
pads = detector.epix100_pad_geometry_list(detector_distance=detector_distance)
# Find the appropriate spacegroup string by doing this: print(crystal.get_hm_symbols())
uc = (4.176e-10, 4.721e-10, 58.33e-10, 89.44*deg, 89.63*deg, 75.85*deg)
cryst = crystal.CrystalStructure(pdb_id, spacegroup='P -1', unitcell=uc)
cryst.crystal_size = crystal_size
cryst.mosaic_domain_size = crystal_domain_size
class AgBFrameGetter(FrameGetter):
    def __init__(self, pads, beam, cryst, poisson=False):
        super().__init__()
        self.n_frames = int(1e6)
        self.pads = pads
        self.beam = beam
        self.cryst = cryst
        self.poisson = poisson
        q_mags = pads.q_mags(beam=beam)
        q_vecs = pads.q_vecs(beam=beam)
        to_photons = r_e**2
        to_photons *= pads.solid_angles()
        to_photons *= pads.polarization_factors(beam=beam)
        to_photons *= beam.photon_number_fluence
        self.to_photons = to_photons
        simcore = clcore.ClCore()
        self.simcore = simcore
        self.q_vecs_gpu = simcore.to_device(q_vecs)
        self.amps_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.complex_t)
        self.amps_tmp_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.complex_t)
        self.I_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.real_t)
        self.S2_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.real_t)
        self.F2_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.real_t)
        r_vecs = cryst.get_symmetry_expanded_coordinates()
        r_vecs -= np.mean(r_vecs, axis=0)
        # For speed: Group position vectors according to atom type to avoid re-calculating scattering factors.
        # For speed: Allocate all GPU arrays upfront go avoid unnecessary data transfers.
        uniq_z = np.unique(cryst.molecule.atomic_numbers)
        grouped_r_vecs_gpu = []
        grouped_fs_gpu = []
        grouped_ones_gpu = []
        for z in uniq_z:
            f = atoms.hubbel_henke_scattering_factors(q_mags=q_mags, photon_energy=beam.photon_energy, atomic_number=z)
            r = utils.atleast_2d(np.squeeze(r_vecs[np.where(cryst.molecule.atomic_numbers == z), :]))
            grouped_r_vecs_gpu.append(simcore.to_device(r))
            grouped_fs_gpu.append(simcore.to_device(f))
            grouped_ones_gpu.append(simcore.to_device(shape=(r.shape[0],), dtype=simcore.complex_t) * 0 + 1)
        self.grouped_r_vecs_gpu = grouped_r_vecs_gpu
        self.grouped_fs_gpu = grouped_fs_gpu
        self.grouped_ones_gpu = grouped_ones_gpu
    def get_data(self, frame_number=1):
        np.random.seed(int(frame_number*2022))
        R = Rotation.random().as_matrix()
        # The intensities are I(q) = |F(q)|^2 * |S(q)|^2.
        # First we calculate the molecular transform |F(q)|^2
        cryst = self.cryst
        amps_tmp_gpu = self.amps_tmp_gpu
        amps_gpu = self.amps_gpu*0
        I_gpu = self.I_gpu
        S2_gpu = self.S2_gpu
        F2_gpu = self.F2_gpu
        q_vecs_gpu = self.q_vecs_gpu
        # Do the sum over each atom species
        for j in range(len(self.grouped_fs_gpu)):
            f_gpu = self.grouped_fs_gpu[j]
            r_gpu = self.grouped_r_vecs_gpu[j]
            ones_gpu = self.grouped_ones_gpu[j]
            self.simcore.phase_factor_qrf(q_vecs_gpu, r_gpu, f=ones_gpu, R=R, a=amps_tmp_gpu, add=False)
            amps_gpu += amps_tmp_gpu * f_gpu
        self.simcore.mod_squared_complex_to_real(amps_gpu, F2_gpu)
        # Now we calculate the shape transform |S(q)|^2.
        # The crystal basis vectors are needed for the shape transform
        abc = cryst.unitcell.o_mat.T.copy()
        # Number of unit cells along each basis vector
        Nfull = cryst.crystal_size / np.array([cryst.unitcell.a, cryst.unitcell.b, cryst.unitcell.c])
        # If the crystal is too big, the shape transform is so tiny that we never even sample it on our relatively
        # sparse grid of pixels!!! So we need to artificially broaden the peaks, while maintaining the correct overall
        # intensity.  We will simulate the intensities from a smaller domain, and then sum up intensities from all
        # domains.  This is also a crude partially coherent crystal mosaic domain model.
        Ndomain = cryst.mosaic_domain_size / np.array([cryst.unitcell.a, cryst.unitcell.b, cryst.unitcell.c])
        Ndomain[Ndomain < 1] = 1
        n_domains = max(np.product(Nfull) / float(np.product(Ndomain)), 1)
        N = Ndomain.astype(int)
        self.simcore.gaussian_lattice_transform_intensities(q_vecs_gpu, abc, N, R=R, I=S2_gpu, add=False)
        S2_gpu *= n_domains
        I_gpu = S2_gpu * F2_gpu
        # Finally, the intensities go from CPU to GPU
        I = I_gpu.get()
        I *= self.to_photons
        if self.poisson:
            I = np.double(np.random.poisson(I))
        df = dataframe.DataFrame()
        df.set_beam(beam)
        df.set_pad_geometry(pads)
        df.set_raw_data(I)
        return df
# For displaying data, we should pre-process to convert to log scale.
def do_logscale(df):
    a = df.get_raw_data_flat()
    a = np.log10(a+1)
    df.set_processed_data(a)
    return df
fg = AgBFrameGetter(pads, beam, cryst, poisson=add_poisson_noise)
from time import time
t = time()
for i in range(1000):
    df = fg.get_next_frame()
print((time()-t)/1000, 'seconds per pattern')
# fg.view(dataframe_preprocessor=do_logscale, percentiles=[0, 10]) #preprocessor) #, levels=[-1, 10])
pv = fg.get_padview(dataframe_preprocessor=do_logscale, percentiles=[0, 10])
d = 58.3385e-10
pv.add_rings(d_spacings=d/(np.arange(2)+1))
pv.start()
# Sanity check: does the forward scatter equal the square of the number of electrons?

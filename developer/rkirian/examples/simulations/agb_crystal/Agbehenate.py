"""
Simulate Ag Behenate diffraction patterns from protein pdb_id on a Jungfrau 4M detector.

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
#------------------------------
# np.random.seed(2021)    # Make random numbers that are reproducible
photon_energy = 9500     # eV
diameter_fwhm = 1e-6      # m
pulse_energy = 1e-3       # J
detector_distance = 0.5   # m
pdb_id = '1507774.pdb'    # Hacked AgB "pdb" file
N_patterns = 10           # Number of patterns to simulate
crystal_size = 100e-9     # Overall crystal size (affects integrated Bragg peak intensity)
crystal_domain_size = min(crystal_size, 100e-9)  # Domain size of crystal (affects Bragg peak size)
show_patterns = 1
add_poisson_noise = 0     # Turn Poisson noise on or off
#------------------------------
eV = const.eV
r_e = const.r_e
beam = source.Beam(photon_energy=photon_energy*eV, diameter_fwhm=diameter_fwhm, pulse_energy=pulse_energy)
fluence = beam.photon_number_fluence
pads = detector.epix100_pad_geometry_list(detector_distance=detector_distance)
pads = detector.PADGeometryList(pads[:4])  # why does slicing return list instead of PADGeometryList?
pads = pads.binned(1)
q_vecs = pads.q_vecs(beam=beam)
solid_angles = pads.solid_angles()
polarization_factors = pads.polarization_factors(beam=beam)
q_mags = pads.q_mags(beam=beam)
# Find the appropriate spacegroup string by doing this:
hms = crystal.get_hm_symbols()
print(hms[0:5])
# spacegroup = crystal.SpaceGroup(hermann_mauguin_symbol='P -1')
unitcell = crystal.UnitCell(a=4.1769e-10, b=4.7218e-10, c=58.3385e-10,
                            alpha=89.440/180*np.pi, beta=89.634/180*np.pi, gamma=75.854/180 * np.pi)
cryst = crystal.CrystalStructure(pdb_id, spacegroup='P -1', unitcell=unitcell)
# Sanity check: the forward scatter molecular transofrm is equal to # electrons squared.  Good.
# print('ztot2', np.sum(cryst.molecule.atomic_numbers)**2)
r_vecs = cryst.get_symmetry_expanded_coordinates()  # molecule.coordinates
r_vecs -= np.mean(r_vecs, axis=0)
simcore = clcore.ClCore() #double_precision=True)
# For speed: group position vectors according to atom type, which helps avoid re-calculating scattering factors
# For speed: allocate all GPU arrays upfront go avoid data transfers
uniq_z = np.unique(cryst.molecule.atomic_numbers)
grouped_r_vecs_gpu = []
grouped_fs_gpu = []
grouped_ones_gpu = []
for z in uniq_z:
    subr = r_vecs[np.where(cryst.molecule.atomic_numbers == z), :]
    subr = utils.atleast_2d(np.squeeze(subr))
    grouped_r_vecs_gpu.append(simcore.to_device(subr))
    f = atoms.hubbel_henke_scattering_factors(q_mags=q_mags, photon_energy=beam.photon_energy, atomic_number=z)
    grouped_fs_gpu.append(simcore.to_device(f))
    grouped_ones_gpu.append(simcore.to_device(shape=(subr.shape[0],),  dtype=simcore.complex_t)*0 + 1)
q_vecs_gpu = simcore.to_device(q_vecs)
I_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.real_t)
amps_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.complex_t)
amps_tmp_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.complex_t)
S2_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.real_t)
F2_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.real_t)
f_ones_gpu = simcore.to_device(shape=(q_vecs.shape[0],),  dtype=simcore.complex_t)*0 + 1
class AgBFrameGetter(FrameGetter):
    def __init__(self):
        self.n_frames = int(1e6)
        self.amps_gpu = amps_gpu
        self.amps_tmp_gpu = amps_tmp_gpu
        self.I_gpu = I_gpu
        self.S2_gpu = S2_gpu
    def get_data(self, frame_number=1):
        np.random.seed(int(frame_number*2022))
        R = Rotation.random().as_matrix()
        # The intensities are I(q) = |F(q)|^2 * |S(q)|^2.
        # First we calculate the molecular transform |F(q)|^2
        amps_tmp_gpu = self.amps_tmp_gpu
        amps_gpu = self.amps_gpu*0
        I_gpu = self.I_gpu*0
        S2_gpu = self.S2_gpu*0
        # Do the sum over each atom species
        for j in range(len(grouped_fs_gpu)):
            f_gpu = grouped_fs_gpu[j]
            r_gpu = grouped_r_vecs_gpu[j]
            ones_gpu = grouped_ones_gpu[j]
            simcore.phase_factor_qrf(q_vecs_gpu, r_gpu, f=ones_gpu, R=R, a=amps_tmp_gpu, add=False)
            amps_gpu += amps_tmp_gpu * f_gpu
        simcore.mod_squared_complex_to_real(amps_gpu, F2_gpu)
        # Now we calculate the shape transform |S(q)|^2.
        # The crystal basis vectors are needed for the shape transform
        abc = cryst.unitcell.o_mat.T.copy()
        # Number of unit cells along each basis vector
        Nfull = crystal_size / np.array([cryst.unitcell.a, cryst.unitcell.b, cryst.unitcell.c])
        # If the crystal is too big, the shape transform is so tiny that we never even sample it on our relatively sparse
        # grid of pixels!!! So we need to artificially broaden the peaks, while maintaining the correct overall
        # intensity.  We will simulate the intensities from a smaller domain, and then sum up intensities from all domains.
        # This is also a crude partially coherent crystal mosaic domain model.
        Ndomain = crystal_domain_size / np.array([cryst.unitcell.a, cryst.unitcell.b, cryst.unitcell.c])
        Ndomain[Ndomain < 1] = 1
        n_domains = max(np.product(Nfull) / float(np.product(Ndomain)), 1)
        N = Ndomain.astype(int)
        simcore.gaussian_lattice_transform_intensities(q_vecs_gpu, abc, N, R=R, I=S2_gpu, add=False)
        S2_gpu *= n_domains
        I_gpu = S2_gpu * F2_gpu
        # Finally, the intensities go from CPU to GPU
        I = I_gpu.get()
        I *= r_e ** 2 * fluence * solid_angles * polarization_factors
        if add_poisson_noise:
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
fg = AgBFrameGetter()
# fg.view(dataframe_preprocessor=do_logscale, percentiles=[0, 10]) #preprocessor) #, levels=[-1, 10])
pv = fg.get_padview(dataframe_preprocessor=do_logscale, percentiles=[0, 10])
d = 58.3385e-10
pv.add_rings(d_spacings=d/(np.arange(2)+1))
pv.start()

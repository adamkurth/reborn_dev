# This file has some scraps of code that were once a part of reborn, but which were removed in favor of improved or
# more elegant code.
#
# @jit(nopython=True)
# def trilinear_interpolation_numba(densities=None, vectors=None, corners=None, deltas=None, out=None):
#     r"""
#     Trilinear interpolation of a 3D map.
#
#     Arguments:
#         densities: A 3D array of shape AxBxC
#         vectors: An Nx3 array of 3-vectors
#         limits: A 3x2 array specifying the limits of the density map samples.  These values specify the voxel centers.
#
#     Returns: Array of intensities with length N.
#     """
#
#     nx = int(densities.shape[0])
#     ny = int(densities.shape[1])
#     nz = int(densities.shape[2])
#
#     for ii in range(vectors.shape[0]):
#
#         # Floating point coordinates
#         i_f = float(vectors[ii, 0] - corners[0, 0]) / deltas[0]
#         j_f = float(vectors[ii, 1] - corners[1, 0]) / deltas[1]
#         k_f = float(vectors[ii, 2] - corners[2, 0]) / deltas[2]
#
#         # Integer coordinates
#         i = int(np.floor(i_f)) % nx
#         j = int(np.floor(j_f)) % ny
#         k = int(np.floor(k_f)) % nz
#
#         # Trilinear interpolation formula specified in e.g. paulbourke.net/miscellaneous/interpolation
#         k0 = k
#         j0 = j
#         i0 = i
#         k1 = k+1
#         j1 = j+1
#         i1 = i+1
#         x0 = i_f - np.floor(i_f)
#         y0 = j_f - np.floor(j_f)
#         z0 = k_f - np.floor(k_f)
#         x1 = 1.0 - x0
#         y1 = 1.0 - y0
#         z1 = 1.0 - z0
#         out[ii] = densities[i0, j0, k0] * x1 * y1 * z1 + \
#                   densities[i1, j0, k0] * x0 * y1 * z1 + \
#                   densities[i0, j1, k0] * x1 * y0 * z1 + \
#                   densities[i0, j0, k1] * x1 * y1 * z0 + \
#                   densities[i1, j0, k1] * x0 * y1 * z0 + \
#                   densities[i0, j1, k1] * x1 * y0 * z0 + \
#                   densities[i1, j1, k0] * x0 * y0 * z1 + \
#                   densities[i1, j1, k1] * x0 * y0 * z0
#
#     return out
#
# @jit(nopython=True)
# def place_atoms_in_map(x_vecs, atom_fs, sigma, s, orth_mat, map_x_vecs, f_map, f_map_tmp):
#
#         r"""
#
#         Needs documentation...
#
#         """
#
#         n_atoms = x_vecs.shape[0]
#         n_map_voxels = map_x_vecs.shape[0]
#         # f_map = np.empty([n_map_voxels], dtype=atom_fs.dtype)
#         # f_map_tmp = np.empty([n_map_voxels], dtype=x_vecs.dtype)
#         for n in range(n_atoms):
#             x = x_vecs[n, 0] % s
#             y = x_vecs[n, 1] % s
#             z = x_vecs[n, 2] % s
#             w_tot = 0
#             for i in range(n_map_voxels):
#                 mx = map_x_vecs[i, 0]
#                 my = map_x_vecs[i, 1]
#                 mz = map_x_vecs[i, 2]
#                 dx = np.abs(x - mx)
#                 dy = np.abs(y - my)
#                 dz = np.abs(z - mz)
#                 dx = min(dx, s - dx)
#                 dy = min(dy, s - dy)
#                 dz = min(dz, s - dz)
#                 dr2 = (orth_mat[0, 0] * dx + orth_mat[0, 1] * dy + orth_mat[0, 2] * dz)**2 + \
#                       (orth_mat[1, 0] * dx + orth_mat[1, 1] * dy + orth_mat[1, 2] * dz)**2 + \
#                       (orth_mat[2, 0] * dx + orth_mat[2, 1] * dy + orth_mat[2, 2] * dz)**2
#                 w = np.exp(-dr2/(2*sigma**2))
#                 f_map_tmp[i] = w
#                 w_tot += w
#             f_map += atom_fs[n] * f_map_tmp/w_tot
#
# @jit(['void(float64[:], float64[:], float64[:], float64[:], float64[:])'], nopython=True)
# def trilinear_insertion(densities=None, weights=None, vectors=None, input_densities=None, limits=None):
#     r"""
#     Trilinear "insertion" -- basically the opposite of trilinear interpolation.  This places densities into a grid
#     using the same weights as in trilinear interpolation.
#
#     Arguments:
#         densities (NxMxP array):
#         weights (NxMxP array):
#         vectors (Qx3 array):
#         input_densities (length-Q array):
#         limits (3x2 array): A 3x2 array specifying the limits of the density map samples.  These values specify the
#                             voxel centers.
#
#     Returns: None -- the inputs densities and weights are modified by this function
#     """
#
#     nx = int(densities.shape[0])
#     ny = int(densities.shape[1])
#     nz = int(densities.shape[2])
#
#     dx = (limits[0, 1] - limits[0, 0]) / nx
#     dy = (limits[1, 1] - limits[1, 0]) / ny
#     dz = (limits[2, 1] - limits[2, 0]) / nz
#
#     for ii in range(vectors.shape[0]):
#
#         # Floating point coordinates
#         i_f = float(vectors[ii, 0] - limits[0, 0]) / dx
#         j_f = float(vectors[ii, 1] - limits[1, 0]) / dy
#         k_f = float(vectors[ii, 2] - limits[2, 0]) / dz
#
#         # Integer coordinates
#         i = int(np.floor(i_f))
#         j = int(np.floor(j_f))
#         k = int(np.floor(k_f))
#
#         # Trilinear interpolation formula specified in e.g. paulbourke.net/miscellaneous/interpolation
#         k0 = k
#         j0 = j
#         i0 = i
#         k1 = k+1
#         j1 = j+1
#         i1 = i+1
#         x0 = i_f - np.floor(i_f)
#         y0 = j_f - np.floor(j_f)
#         z0 = k_f - np.floor(k_f)
#         x1 = 1.0 - x0
#         y1 = 1.0 - y0
#         z1 = 1.0 - z0
#         if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
#             val = input_densities[ii]
#             densities[i0, j0, k0] += val
#             densities[i1, j0, k0] += val
#             densities[i0, j1, k0] += val
#             densities[i0, j0, k1] += val
#             densities[i1, j0, k1] += val
#             densities[i0, j1, k1] += val
#             densities[i1, j1, k0] += val
#             densities[i1, j1, k1] += val
#             weights[i0, j0, k0] += x1 * y1 * z1
#             weights[i1, j0, k0] += x0 * y1 * z1
#             weights[i0, j1, k0] += x1 * y0 * z1
#             weights[i0, j0, k1] += x1 * y1 * z0
#             weights[i1, j0, k1] += x0 * y1 * z0
#             weights[i0, j1, k1] += x1 * y0 * z0
#             weights[i1, j1, k0] += x0 * y0 * z1
#             weights[i1, j1, k1] += x0 * y0 * z0


# from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

# import refdata


# def amplitudes_with_cmans(q, r, Z):
#     """
#     compute scattering amplitudes
#
#     q: 2D np.array of q vectors
#     r: 2D np.array of atom coors
#     Z: 1D np.array of atomic numbers corresponding to r
#
#     FIXME: Derek - is this used anywhere?
#     """
#
#     cman = refdata.get_cromermann_parameters(Z)
#     form_facts = refdata.get_cmann_form_factors(cman, q)
#     ff_mat = np.array([form_facts[z] for z in Z]).T.astype(np.float32)
#     amps = (np.dot(q, r.T)).astype(np.float32)
#     amps = np.exp(1j * amps).astype(np.complex64)
#     amps = np.sum(amps * ff_mat, 1).astype(np.complex64)
#     return amps
#
#
# def amplitudes(q, r):
#     """
#     compute scattering amplitudes without form factors
#
#     q: 2D np.array of q vectors
#     r: 2D np.array of atom coors
#
#     FIXME: Derek - is this used anywhere?
#     """
#     amps = np.dot(q, r.T)
#     amps = np.exp(1j * amps)
#     amps = np.sum(amps, 1)
#     return amps


# def sphericalize(lattice):
#     """attempt to sphericalize a 2D lattice point array"""
#     center = lattice.mean(0)
#     rads = np.sqrt(np.sum((lattice - center) ** 2, 1))
#     max_rad = min(lattice.max(0)) / 2.
#     return lattice[rads < max_rad]


# def area_beam(lb, r):
#     '''
#     This function returns the cross sectional area of illuminated
#     solvent assuming no crystal is present. if the beam side length
#     is greater than 2r it reutrns the cross-sectional area of the jet.
#
#     arguments
#     ------------
#     lb: the side length of the beam
#     r:  the radius of the jet
#
#     returns
#     ------------
#     float value of the cross sectional area of illuminated jet
#     '''
#
#     if lb >= 2 * r:
#         return np.pi * r * r
#     elif lb > 0 and lb < 2 * r:
#         x = np.sqrt(4 * r * r - lb * lb) / 2
#         theta = np.arccos(lb / (2 * r))
#         sliver = (theta * r * r) - (x * lb / 2)
#         return np.pi * r * r - 2 * sliver
#     else:
#         print("beam side length must be greater than zero.")


# def area_crys(lc, r):
#     '''
#     This function finds the exposed cross-sectional area of solvent
#     from a cubic crystal and cylindrical jet. If the crystal is larger
#     than the beam it returns zero. Its mainly used in the function
#     "volume_solvent."
#
#     arguments
#     -------------
#     lc: the side length of the crystal
#     r:  the radius of the jet
#
#     returns
#     -------------
#     a float value for the cross-sectional area of illuminated solvent
#     '''
#
#     if lc >= 2 * r:
#         return 0
#     elif lc > np.sqrt(2) * r:
#         x = np.sqrt(4 * r * r - lc * lc) / 2
#         theta = np.arccos(lc / (2 * r))
#         sliver = (theta * r * r) - (x * lc / 2)
#         return 4 * sliver
#     elif lc > 0 and lc <= np.sqrt(2) * r:
#         return np.pi * r * r - lc * lc
#     else:
#         print("crystal side length must be greater than zero.")


# def volume_solvent(lb, lc, r):
#     '''
#     This function returns the volume of illuminated solvent assuming
#     a "square" beam, a cubic crystal and a cylindrical jet of solvent.
#
#     arguments
#     --------------
#     lb: The side length of the square beam
#     lc: The side length of the cubuc crystal
#     r:  The radius of the solvent jet
#
#     returns
#     --------------
#     float value of the volume of illuminated solvent
#     '''
#
#     if lb >= 2 * r:
#         if lc >= lb:
#             return 0
#         elif lc >= 2 * r:
#             return np.pi * r * r * (lb - lc)
#         elif lc > np.sqrt(2) * r:
#             return np.pi * r * r * (lb - lc) + lc * (area_crys(lc, r))
#         elif lc > 0:
#             return np.pi * r * r * (lb) - lc**3
#         else:
#             print("Crystal side length must be greater than zero")
#     elif lb > 0:
#         if lc >= 2 * r:
#             return 0
#         elif lc >= np.sqrt(2) * r:
#             if lc >= lb:
#                 x = np.sqrt(4 * r * r - lc * lc)
#                 if lb >= x:
#                     return area_crys(lc, r) * lb / 2
#                 elif lb < x:
#                     return (area_beam(lb, r) - (lb * lc)) * lb
#             if lb > lc:
#                 return (area_crys(lc, r) +
#                         area_beam(lb, r) - np.pi * r * r) * lb
#         elif lc > 0:
#             if lc >= lb:
#                 return (area_beam(lb, r) - lb * lc) * lb
#             else:
#                 return area_beam(lb, r) * lb - lc**3
#         else:
#             print("Crystal side length must be greater than zero.")

# Undocumented stuff gets commented out
# Do water scattering
# def simulate_water_jet_background(beam_diameter, crystal_size, water_radius):
#     illuminated_water_volume = simutils.volume_solvent(beam_diameter, crystal_size, water_radius)
#     F_water = solutions.get_water_profile(qmag, temperature=temperature)
#     F2_water = F_water**2 * n_water_molecules
#     I_water = I0 * r_e**2 * pol * sa * F2_water
#     if(illuminated_water_volume <= 0):
#         write('\nWarning: No solvent was illuminated, water scattering not performed.\n')
#         I_water = 0
#     else:
#         write('done\n')
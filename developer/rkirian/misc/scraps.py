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

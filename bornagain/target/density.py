from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from bornagain.target import crystal
import scipy
from scipy.stats import binned_statistic_dd
from numba import jit


class DensityMap3D(object):

    density = None

    def __init__(self, shape=None, centers=None):

        if self.shape is None:
            raise ValueError("Shape must have three integer values")




class CrystalMeshTool(object):

    r'''
    A helper class for working with 3D density maps.  Most importantly, it allows one to do spacegroup symmetry
    transformations.  Once provided information about a crystal (spacegroup, lattice), along with desired resolution and
    oversampling ratio, this tool intelligently chooses the shape of the map and creates lookup tables for the symmetry
    transformations.  It also produces the Fourier-space frequency samples that are helpful when connecting direct
    all-atom simulations with FFTs.  In general, the maps are cubic NxNxN array.  This class does not maintain the data
    array; it provides methods needed to work on the data arrays.

    Importantly, this class is focused on operations in the crystal basis, because the symmetry operations are most
    elegant in that basis.
    '''

    sym_luts = None  #:  Cached lookup tables
    # n_vecs = None
    # x_vecs = None
    # r_vecs = None
    # h_vecs = None

    def __init__(self, cryst, resolution, oversampling):

        r'''
        This should intelligently pick the limits of a map.  On initialization, you only need to provide a
        target.crystal.Structure() class instance, along with your desired resolution and oversampling.  You can create
        the target.crystal.Structure() class most easily if you have a pdb file as follows:

        Arguments:
            cryst (crystal.Structure) : A crystal structure that contains the spacegroup and lattice information.
            resolution (float) : The desired resolution of the map (will be modified to suit integer samples and a
                                  square 3D mesh)
            oversampling (int) : An oversampling of 2 gives a real-space map that is twice as large as the unit cell. In
                                  Fourier space, there will be one sample between Bragg samples.  And so on for 3,4,...
        '''

        d = resolution
        s = np.ceil(oversampling)

        abc = np.array([cryst.unitcell.a, cryst.unitcell.b, cryst.unitcell.c])

        m = 1
        for T in cryst.spacegroup.sym_translations:
            for mp in np.arange(1, 10):
                Tp = T*mp
                if np.round(np.max(Tp % 1.0)*100)/100 == 0:
                    if mp > m:
                        m = mp
                    break

        # The idea here is that if, for example, we have a 3-fold screw axis in the spacegroup, then the number of
        # samples in the map should be a multiple of 3.  Hopefully this works...
        Nc = np.max(np.ceil(abc / (d * m)) * m)

        self.cryst = cryst  #:  crystal.target class object used to initiate the map
        # self.m = np.int(m)  #:  Due to symmetry translations, map must consit of integer multiple of this number
        self.s = np.int(s)  #:  Oversampling ratio
        self.d = abc / Nc   #:  Length-3 array of "actual" resolutions (different from "requested" resolution)
        self.dx = 1 / Nc    #:  Crystal basis length increment
        self.Nc = np.int(Nc)  #:  Number of samples along edge of unit cell
        self.N = np.int(Nc * s)  #:  Number of samples along edge of whole map (includes oversampling)
        self.P = np.int(self.N**3)  #:  Linear length fo map (=N^3)
        self.w = np.array([self.N**2, self.N, 1])  #:  The stride vector (mostly for internal use)

    def get_n_vecs(self):

        r"""

        Get an Nx3 array of vectors corresponding to the indices of the map voxels.  The array looks like this:

        [[0,0,0],[0,0,1],[0,0,2], ... ,[N-1,N-1,N-1]]

        Returns: numpy array

        """

        P = self.P
        N = self.N
        p = np.arange(0, P)
        n_vecs = np.zeros([P, 3])
        n_vecs[:, 0] = np.floor(p / (N**2))
        n_vecs[:, 1] = np.floor(p / N) % N
        n_vecs[:, 2] = p % N
        # n_vecs = n_vecs[:, ::-1]
        return n_vecs

    def get_x_vecs(self):

        r"""

        Get an Nx3 array of vectors in the crystal basis.  If there were four samples per unit cell, the array looks a
        bit like this:

        [[0,0,0],[0,0,0.25],[0,0,0.5],[0,0,0.75],[0,0.25,0], ... ,[0.75,0.75,0.75]]

        Returns: numpy array

        """

        x_vecs = self.get_n_vecs()
        x_vecs = x_vecs * self.dx
        return x_vecs

    def get_r_vecs(self):

        r"""

        Creates an Nx3 array of 3-vectors contain the Cartesian-basis vectors of each voxel in the map.  This is done
        by taking the crystal-basis position vectors x, and applying the orthogonalization matrix O to them.

        Returns: numpy array

        """

        x = self.get_x_vecs()

        return np.dot(self.cryst.unitcell.o_mat, x.T).T

    def get_h_vecs(self):

        r"""

        This provides an Nx3 array of Fourier-space vectors "h".  These coordinates can be understood as "fractional
        Miller indices" that coorespond to the density samples upon taking an FFT of the real-space map.  With atomic
        coordinates x (in the crystal basis) one can take the Fourier transform F(h) = sum_n f_n exp(i h*x)

        Returns: numpy array

        """

        h = self.get_n_vecs()
        h = h / self.dx / self.N
        f = np.where(h.ravel() > (self.Nc/2))
        h.flat[f] = h.flat[f] - self.Nc
        return h

    def get_q_vecs(self):

        r"""

		This provides an Nx3 array of momentum-transfer vectors (with the usual 2 pi factor).

		Returns: numpy array

		"""

        return 2*np.pi*np.dot(self.cryst.unitcell.a_mat, self.get_h_vecs().T).T


    def get_sym_luts(self):

        r"""

        This provides a list of "symmetry transform lookup tables".  These are the linearized array indices. For a
        transformation that consists of an identity matrix along with zero translation, the lut is just an array
        p = [0,1,2,3,...,N^3-1].  Other transforms are like a "scrambling" of that ordering, such that a remapping of
        density samples is done with an operation like this: newmap.flat[p2] = oldmap.flat[p1].   Note that the luts are
        kept in memory for future use - beware of the memory requirement.

        Returns: list of numpy arrays

        """

        if self.sym_luts is None:

            sym_luts = []
            x0 = self.get_x_vecs()

            for (R, T) in zip(self.cryst.spacegroup.sym_rotations, self.cryst.spacegroup.sym_translations):
                lut = np.dot(R, x0.T).T + T    # transform x vectors in 3D grid
                lut = np.round(lut / self.dx)  # switch from x to n vectors
                lut = lut % self.N             # wrap around
                lut = np.dot(self.w, lut.T)    # in p space
                sym_luts.append(lut.astype(np.int))
            self.sym_luts = sym_luts

        return self.sym_luts

    def symmetry_transform(self, i, j, data):

        r"""

        This applies symmetry transformations to a data array (i.e. density map).  The crystal spacegroup gives rise
        to Ns symmetry transformation operations (e.g. each operation consists of a rotation matrix paired with
        translation vector).  Of those Ns symmetry transformations, this function will take a map from the ith
        configuration to the jth configuration.

        Arguments:
            i (int) : The "from" index; symmetry transforms are performed from this index to the j index
            j (int) : The "to" index; symmetry transforms are performed from the i index to this index

        Returns: A transformed data array (i.e. density map)


        """

        luts = self.get_sym_luts()
        data_trans = np.zeros_like(data)
        data_trans.flat[luts[j]] = data.flat[luts[i]]

        return data_trans

    def shape(self):

        r"""

        For convenience, return the map shape [N,N,N]

        Returns: numpy array

        """

        return np.array([self.N, self.N, self.N])

    def reshape(self, data):

        r"""

        For convenience, this will reshape a data array to the shape NxNxN.

        Args:
            data: the data array

        Returns: the same data array as the input, but with shape NxNxN

        """

        N = self.N
        return data.reshape([N, N, N])

    def reshape3(self, data):

        r"""

        Args:
            data: A data array of length 3*N^3 that consists of 3-vectors (one for each density voxel)

        Returns: A re-shaped data array of shape NxNxNx3

        """

        N = self.N
        return data.reshape([N, N, N, 3])

    def zeros(self):

        r"""

        A convenience function: simply returns an array of zeros of shape NxNxN

        Returns: numpy array

        """

        N = self.N
        return np.zeros([N, N, N])

    def place_atoms_in_map(self, atom_x_vecs, atom_fs, mode='gaussian', fixed_atom_sigma=0.5e-10):

        r"""

        This will take a list of atom position vectors and densities and place them in a 3D map.  The position vectors
        should be in the crystal basis, and the densities must be real (because the scipy function that we use does
        not allow for complex numbers...).  This is done in a lazy way - the density samples are placed in the nearest
        voxel.  There are no Gaussian shapes asigned to the atomic form.  Nothing fancy...

        Args:
            x (numpy array):  An nx3 array of position vectors
            f (numpy array):  An n-length array of densities (must be real)

        Returns: An NxNxN numpy array containing the sum of densities that were provided as input.

        """
        if mode == 'gaussian':
            sigma = fixed_atom_sigma # Gaussian sigma (i.e. atom "size"); this is a fudge factor and needs to be updated
            # n_atoms = atom_x_vecs.shape[0]
            orth_mat = self.cryst.unitcell.o_mat.copy()
            map_x_vecs = self.get_x_vecs()
            n_map_voxels = map_x_vecs.shape[0]
            f_map = np.zeros([n_map_voxels], dtype=np.complex)
            f_map_tmp = np.zeros([n_map_voxels], dtype=np.double)
            s = self.s
            place_atoms_in_map(atom_x_vecs, atom_fs, sigma, s, orth_mat, map_x_vecs, f_map, f_map_tmp)
            return self.reshape(f_map)
        elif mode == 'nearest':
            mm = [0, self.s]
            rng = [mm, mm, mm]
            a, _, _ = binned_statistic_dd(x, f, statistic='sum', bins=[self.N] * 3, range=rng)
            return a

    def place_intensities_in_map(self, h, f):

        r"""

        This will take a list of Miller index vectors and intensities and place them in a 3D map.

        Args:
            h (numpy array):  An Nx3 array of hkl vectors
            f (numpy array):  An N-length array of intensities (must be real)

        Returns: An NxNxN numpy array containing the sum of densities that were provided as input.

        """

        hh = self.get_h_vecs()
        mm = [np.min(hh)-0.5, np.max(hh)+0.5]
        rng = [mm, mm, mm]
        a, _, _ = binned_statistic_dd(h, f, statistic='mean', bins=[self.N] * 3, range=rng)

        return a


@jit(nopython=True)
def place_atoms_in_map(x_vecs, atom_fs, sigma, s, orth_mat, map_x_vecs, f_map, f_map_tmp):

        r"""

        Needs documentation...

        """

        n_atoms = x_vecs.shape[0]
        n_map_voxels = map_x_vecs.shape[0]
        # f_map = np.empty([n_map_voxels], dtype=atom_fs.dtype)
        # f_map_tmp = np.empty([n_map_voxels], dtype=x_vecs.dtype)
        for n in range(n_atoms):
            x = x_vecs[n, 0] % s
            y = x_vecs[n, 1] % s
            z = x_vecs[n, 2] % s
            w_tot = 0
            for i in range(n_map_voxels):
                mx = map_x_vecs[i, 0]
                my = map_x_vecs[i, 1]
                mz = map_x_vecs[i, 2]
                dx = np.abs(x - mx)
                dy = np.abs(y - my)
                dz = np.abs(z - mz)
                dx = min(dx, s - dx)
                dy = min(dy, s - dy)
                dz = min(dz, s - dz)
                dr2 = (orth_mat[0, 0] * dx + orth_mat[0, 1] * dy + orth_mat[0, 2] * dz)**2 + \
                      (orth_mat[1, 0] * dx + orth_mat[1, 1] * dy + orth_mat[1, 2] * dz)**2 + \
                      (orth_mat[2, 0] * dx + orth_mat[2, 1] * dy + orth_mat[2, 2] * dz)**2
                w = np.exp(-dr2/(2*sigma**2))
                f_map_tmp[i] = w
                w_tot += w
            f_map += atom_fs[n] * f_map_tmp/w_tot


@jit(nopython=True)
def trilinear_interpolation(densities=None, vectors=None, limits=None, out=None):
    r"""
    Trilinear interpolation of a 3D map.

    Args:
        densities: A 3D array of shape AxBxC
        vectors: An Nx3 array of 3-vectors
        limits: A 3x2 array specifying the limits of the density map samples.  These values specify the voxel centers.

    Returns: Array of intensities with length N.
    """

    if out is None:
        out = np.zeros(vectors.shape[0], dtype=densities.dtype)

    nx = int(densities.shape[0])
    ny = int(densities.shape[1])
    nz = int(densities.shape[2])

    dx = (limits[0, 1] - limits[0, 0]) / nx
    dy = (limits[1, 1] - limits[1, 0]) / ny
    dz = (limits[2, 1] - limits[2, 0]) / nz

    for ii in range(vectors.shape[0]):

        # Floating point coordinates
        i_f = float(vectors[ii, 0] - limits[0, 0]) / dx
        j_f = float(vectors[ii, 1] - limits[1, 0]) / dy
        k_f = float(vectors[ii, 2] - limits[2, 0]) / dz

        # Integer coordinates
        i = int(np.floor(i_f))
        j = int(np.floor(j_f))
        k = int(np.floor(k_f))

        # Trilinear interpolation formula specified in e.g. paulbourke.net/miscellaneous/interpolation
        k0 = k
        j0 = j
        i0 = i
        k1 = k+1
        j1 = j+1
        i1 = i+1
        x0 = i_f - np.floor(i_f)
        y0 = j_f - np.floor(j_f)
        z0 = k_f - np.floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
            out[ii] = densities[i0, j0, k0] * x1 * y1 * z1 + \
                     densities[i1, j0, k0] * x0 * y1 * z1 + \
                     densities[i0, j1, k0] * x1 * y0 * z1 + \
                     densities[i0, j0, k1] * x1 * y1 * z0 + \
                     densities[i1, j0, k1] * x0 * y1 * z0 + \
                     densities[i0, j1, k1] * x1 * y0 * z0 + \
                     densities[i1, j1, k0] * x0 * y0 * z1 + \
                     densities[i1, j1, k1] * x0 * y0 * z0
        else:
            out[ii] = 0

    return out


# @jit(['void(float64[:], float64[:], float64[:], float64[:], float64[:])'], nopython=True)
# def trilinear_insertion(densities=None, weights=None, vectors=None, input_densities=None, limits=None):
#     r"""
#     Trilinear "insertion" -- basically the opposite of trilinear interpolation.  This places densities into a grid
#     using the same weights as in trilinear interpolation.
#
#     Args:
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

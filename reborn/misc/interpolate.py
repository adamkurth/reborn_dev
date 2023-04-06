# This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
#
# reborn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# reborn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with reborn.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from .. import fortran

def _trilinear_interpolation_fortran(densities, vectors, corners, deltas, out):
    r"""
    This is the wrapper to the corresponding fortran function.  It is not meant to be used directly.  See the
    ``trilinear_interpolation`` function.

    Arguments:
        densities:
        vectors:
        corners:
        deltas:
        out:
    """
    assert vectors.dtype == np.float64
    assert corners.dtype == np.float64
    assert deltas.dtype == np.float64
    assert densities.flags.c_contiguous
    assert vectors.flags.c_contiguous
    assert corners.flags.c_contiguous
    assert deltas.flags.c_contiguous
    assert out.flags.c_contiguous
    assert np.min(deltas) > 0
    if np.iscomplexobj(densities) or np.iscomplexobj(out):
        assert densities.dtype == np.complex128
        assert out.dtype == np.complex128
        fortran.density_f.trilinear_interpolation_complex(densities.T, vectors.T, corners.T, deltas.T, out.T)
    else:
        assert densities.dtype == np.float64
        assert out.dtype == np.float64
        fortran.density_f.trilinear_interpolation(densities.T, vectors.T, corners.T, deltas.T, out.T)


def trilinear_interpolation(densities, vectors, corners=None, deltas=None, x_min=None, x_max=None, out=None,
                            strict_types=True):
    r"""
    Perform a `trilinear interpolation <https://en.wikipedia.org/wiki/Trilinear_interpolation>`__ on a 3D array.  An
    arbitrary set of sample points in the form of an :math:`N\times 3` array may be specified.

    Notes:
        * This function behaves as if the density is periodic; points that lie out of bounds will wrap around.  This
          might change in the future, in which case a keyword argument will be added so that you may explicitly decide
          what to do in the case of points that lie outside of the grid.  Note that periodic boundaries avoid the
          need for conditional statements within a for loop, which probably makes the function faster.  For now, if you
          think you have points that lie outside of the grid, consider handling them separately.
        * You may specify the output array, which is useful if you wish to simply add to an existing array that you
          have already allocated.  This can make your code faster and reduce memory.  Beware: the out array is not
          over-written -- the underlying fortran function will *add* to the existing *out* array.
        * Only double precision arrays (both real and complex are allowed) at the fortran level.  You may pass in
          other types, but they will be converted to double (or double complex) before passing to the fortran function.
        * Make sure that all your arrays are c-contiguous.
        * An older version of this code allowed the arguments *corners* and *deltas*.  They are discouraged because
          we aim to standardize on the *x_min* and *x_max* arguments documented below.  They may be removed in the
          future.
        * The shape of the 3D array is inferred from the *densities* argument.

    Arguments:
        densities (numpy array): A 3D density array.
        vectors (numpy array): An Nx3 array of vectors that specify the points to be interpolated.
        x_min (float or numpy array): A 3-element vector specifying the *center* of the corner voxel of the 3D array.
                                      If a float is passed instead, it will be replicated to make a 3D array.
        x_max (float or numpy array): Same as x_min, but specifies the opposite corner, with larger values than x_min.
        out (numpy array): If you don't want the output array to be created (e.g for speed), provide it here.
        strict_types (bool): Set this to False if you don't mind your code being slow due to the need to convert
                             datatypes (i.e. copy arrays) on every function call.  Default: True.

    Returns:
        numpy array
    """
    if (corners is not None) and (deltas is not None):
        corners = np.array(corners).copy()
        deltas = np.array(deltas).copy()
    else:
        if (x_min is None) or (x_max is None):
            raise ValueError('trilinear_interpolation requires the x_min and x_max arguments')
        shape = np.array(densities.shape)
        if len(shape) != 3:
            raise ValueError('trilinear_interpolation requires a 3D densities argument')
        x_min = np.atleast_1d(np.array(x_min))
        x_max = np.atleast_1d(np.array(x_max))
        if len(x_min) == 1:
            x_min = np.squeeze(np.array([x_min, x_min, x_min]))
        if len(x_max) == 1:
            x_max = np.squeeze(np.array([x_max, x_max, x_max]))
        deltas = (x_max - x_min) / (shape - 1)
        corners = x_min
    corners = corners.astype(np.float64)
    deltas = deltas.astype(np.float64)
    if not strict_types:
        if np.iscomplexobj(densities):
            densities = densities.astype(np.complex128)
        else:
            densities = densities.astype(np.float64)
        if out is not None:
            if np.iscomplexobj(out):
                out = out.astype(np.complex128)
            else:
                out = out.astype(np.float64)
    if out is None:
        out = np.zeros(vectors.shape[0], dtype=densities.dtype)
    _trilinear_interpolation_fortran(densities, vectors, corners, deltas, out)
    return out


def _trilinear_insertion_fortran(densities, weights, vectors, insert_vals, corners, deltas):
    r"""
    This is the wrapper to the corresponding fortran function.  It is not meant to be used directly.  See the
    ``trilinear_insertion`` function.

    Args:
        densities:
        weights:
        vectors:
        insert_vals:
        corners:
        deltas:
    """
    assert weights.dtype == np.float64
    assert vectors.dtype == np.float64
    assert corners.dtype == np.float64
    assert deltas.dtype == np.float64
    assert densities.flags.c_contiguous
    assert weights.flags.c_contiguous
    assert vectors.flags.c_contiguous
    assert insert_vals.flags.c_contiguous
    assert corners.flags.c_contiguous
    assert deltas.flags.c_contiguous
    if np.iscomplexobj(densities) or np.iscomplexobj(insert_vals):
        assert densities.dtype == np.complex128
        assert insert_vals.dtype == np.complex128
        fortran.density_f.trilinear_insertion_complex(densities.T, weights.T, vectors.T, insert_vals.T, corners.T,
                                                   deltas.T)
    else:
        assert densities.dtype == np.float64
        assert insert_vals.dtype == np.float64
        fortran.density_f.trilinear_insertion_real(densities.T, weights.T, vectors.T, insert_vals.T, corners.T,
                                                   deltas.T)


def trilinear_insertion(densities, weights, vectors, insert_vals, corners=None, deltas=None, x_min=None, x_max=None):
    r"""
    Perform the "inverse" of a `trilinear interpolation <https://en.wikipedia.org/wiki/Trilinear_interpolation>`__ .
    That is, take an arbitrary set of sample values along with their 3D vector locations and "insert" them into a 3D
    grid of densities.  The values are distributed amongst the nearest 8 grid points so that they sum to the original
    insert value.

    FIXME: Be more clear about the mathematical operation that this function performs...

    Notes:
        * This function behaves as if the density is periodic; points that lie out of bounds will wrap around.  This
          might change in the future, in which case a keyword argument will be added so that you may explicitly decide
          what to do in the case of points that lie outside of the grid.  Note that periodic boundaries avoid the
          need for conditional statements within a for loop, which probably makes the function faster.  For now, if you
          think you have points that lie outside of the grid, consider handling them separately.
        * You may specify the output array, which is useful if you wish to simply add to an existing 3D array that you
          have already allocated.  This can make your code faster and reduce memory.  Beware: the out array is not
          over-written -- the underlying fortran function will *add* to the existing ``densities`` array.
        * Only double precision arrays (both real and complex are allowed).
        * Make sure that all your arrays are c-contiguous.
        * An older version of this code allowed the arguments ``corners`` and ``deltas``.  They are discouraged because
          we aim to standardize on the ``x_min`` and ``x_max`` arguments documented below.  They may be removed in the
          future.
        * The shape of the 3D array is inferred from the ``densities`` argument.

    Arguments:
        densities (numpy array): A 3D array containing the densities, into which values are inserted.  Note that an
                                 "insertion" means that the ``insert_vals`` below are multiplied by ``weights`` below.
        weights (numpy array): A 3D array containing weights.  These are needed in order to perform a weighted average.
                               After calling this function one or more times, the average densities are calculated by
                               dividing ``densities`` by ``weights``.  Be mindful of divide-by-zero errors.
        vectors (numpy array): The 3D vector positions corresponding to the values to be inserted.
        insert_vals (numpy array): The values to be inserted into the 3D map.  They are multiplied by weights before
                                   being inserted into the densities map.
        x_min (float or numpy array): A 3-element vector specifying the *center* of the corner voxel of the 3D array.
                                      If a float is passed instead, it will be replicated to make a 3D array.
        x_max (float or numpy array): Same as x_min, but specifies the opposite corner, with larger values than x_min.

    Returns:
        None.  This function modifies the densities and weights arrays; it returns nothing.
    """
    if (corners is not None) and (deltas is not None):
        corners = np.array(corners).copy()
        deltas = np.array(deltas).copy()
    else:
        if (x_min is None) or (x_max is None):
            raise ValueError('trilinear_insertion requires the x_min and x_max arguments')
        shape = np.array(densities.shape)
        if len(shape) != 3:
            raise ValueError('trilinear_insertion requires a 3D densities argument')
        x_min = np.atleast_1d(np.array(x_min))
        x_max = np.atleast_1d(np.array(x_max))
        if len(x_min) == 1:
            x_min = np.squeeze(np.array([x_min, x_min, x_min]))
        if len(x_max) == 1:
            x_max = np.squeeze(np.array([x_max, x_max, x_max]))
        deltas = (x_max - x_min) / (shape - 1)
        corners = x_min
    corners = corners.astype(np.float64)
    deltas = deltas.astype(np.float64)

    if densities.dtype != np.float64:
        if densities.dtype != np.complex128:
            raise ValueError('trilinear_interpolation requires densities of numpy.float64 or numpy.complex128 type')
    if insert_vals.dtype != np.float64:
        if insert_vals.dtype != np.complex128:
            raise ValueError('trilinear_interpolation requires insert_vals of numpy.float64 or numpy.complex128 type')
    _trilinear_insertion_fortran(densities, weights, vectors, insert_vals, corners, deltas)
    return None


def trilinear_insertion_factor(densities, weight_factor, vectors, insert_vals, corners=None, deltas=None,
                               x_min=None, x_max=None):
    r"""
    Performs trilinear insert with a factor being multiplied onto the weights.

    Notes:
        * This function behaves as if the density is periodic; points that lie out of bounds will wrap around.  This
          might change in the future, in which case a keyword argument will be added so that you may explicitly decide
          what to do in the case of points that lie outside of the grid.  Note that periodic boundaries avoid the
          need for conditional statements within a for loop, which probably makes the function faster.  For now, if you
          think you have points that lie outside of the grid, consider handling them separately.
        * You may specify the output array, which is useful if you wish to simply add to an existing 3D array that you
          have already allocated.  This can make your code faster and reduce memory.  Beware: the out array is not
          over-written -- the underlying fortran function will *add* to the existing ``densities[1,:,:,:]`` array.
        * Make sure that all your arrays are c-contiguous.
        * An older version of this code allowed the arguments ``corners`` and ``deltas``.  They are discouraged because
          we aim to standardize on the ``x_min`` and ``x_max`` arguments documented below.  They may be removed in the
          future.
        * The shape of the 3D array is inferred from the ``densities`` argument.

    Arguments:
        densities (numpy array): A 4D array containing the densities, into which values are inserted.  Note that an
                                 "insertion" means that the ``insert_vals`` below are multiplied by ``weights`` below.
        weight_factor (numpy array): A number that gets multiplied with the trilinear insertion weights.
        vectors (numpy array): The 3D vector positions corresponding to the values to be inserted.
        insert_vals (numpy array): The values to be inserted into the 3D map.  They are multiplied by weights before
                                   being inserted into the densities map.
        x_min (float or numpy array): A 3-element vector specifying the *center* of the corner voxel of the 3D array.
                                      If a float is passed instead, it will be replicated to make a 3D array.
        x_max (float or numpy array): Same as x_min, but specifies the opposite corner, with larger values than x_min.

    Returns:
        None.  This function modifies the densities and weights arrays; it returns nothing.
    """

    # # Convert to f-contiguous arrays required by f2py
    # densities = densities.T
    # vectors = vectors.T

    if (corners is not None) and (deltas is not None):
        corners = np.array(corners).copy()
        deltas = np.array(deltas).copy()
    else:
        if (x_min is None) or (x_max is None):
            raise ValueError('trilinear_insertion requires the x_min and x_max arguments')
        shape = np.array(densities.shape)
        if len(shape) != 4:
            raise ValueError('trilinear_insertion requires a 4D densities argument')
        x_min = np.atleast_1d(np.array(x_min))
        x_max = np.atleast_1d(np.array(x_max))
        if len(x_min) == 1:
            x_min = np.squeeze(np.array([x_min, x_min, x_min]))
        if len(x_max) == 1:
            x_max = np.squeeze(np.array([x_max, x_max, x_max]))

        shape_3D = np.array(densities[:, :, :, 0].shape)
        deltas = (x_max - x_min) / (shape_3D - 1)
        corners = x_min

    corners = corners.astype(np.float64)
    deltas = deltas.astype(np.float64)

    assert densities.flags.c_contiguous
    assert vectors.flags.c_contiguous
    assert insert_vals.flags.c_contiguous
    assert corners.flags.c_contiguous
    assert deltas.flags.c_contiguous

    fortran.density_f.trilinear_insertion_factor_real(densities.T, vectors.T, insert_vals.T, corners.T, deltas.T,
                                               weight_factor)
    return None


def trilinear_insertions(densities, vectors, insert_vals, corners=None, deltas=None, x_min=None, x_max=None):
    r"""
    Performs multiple trilinear inserts

    Notes:
        * This function behaves as if the density is periodic; points that lie out of bounds will wrap around.  This
          might change in the future, in which case a keyword argument will be added so that you may explicitly decide
          what to do in the case of points that lie outside of the grid.  Note that periodic boundaries avoid the
          need for conditional statements within a for loop, which probably makes the function faster.  For now, if you
          think you have points that lie outside of the grid, consider handling them separately.
        * You may specify the output array, which is useful if you wish to simply add to an existing 3D array that you
          have already allocated.  This can make your code faster and reduce memory.  Beware: the out array is not
          over-written -- the underlying fortran function will *add* to the existing ``densities[1,:,:,:]`` array.
        * Make sure that all your arrays are c-contiguous.
        * An older version of this code allowed the arguments ``corners`` and ``deltas``.  They are discouraged because
          we aim to standardize on the ``x_min`` and ``x_max`` arguments documented below.  They may be removed in the
          future.
        * The shape of the 3D array is inferred from the ``densities`` argument.

    Arguments:
        densities (numpy array): A 4D array containing the densities, into which values are inserted.
        vectors (numpy array): The 3D vector positions corresponding to the values to be inserted.
        insert_vals (numpy array): The values to be inserted into the 3D map.  They are multiplied by weights before
                                   being inserted into the densities map.
        x_min (float or numpy array): A 3-element vector specifying the *center* of the corner voxel of the 3D array.
                                      If a float is passed instead, it will be replicated to make a 3D array.
        x_max (float or numpy array): Same as x_min, but specifies the opposite corner, with larger values than x_min.

    Returns:
        None.  This function modifies the densities array; it returns nothing.
    """

    # # Convert to f-contiguous arrays required by f2py
    # densities = densities.T
    # vectors = vectors.T

    if (corners is not None) and (deltas is not None):
        corners = np.array(corners).copy()
        deltas = np.array(deltas).copy()
    else:
        if (x_min is None) or (x_max is None):
            raise ValueError('trilinear_insertion requires the x_min and x_max arguments')
        shape = np.array(densities.shape)
        if len(shape) != 4:
            raise ValueError('trilinear_insertion requires a 4D densities argument')
        x_min = np.atleast_1d(np.array(x_min))
        x_max = np.atleast_1d(np.array(x_max))
        if len(x_min) == 1:
            x_min = np.squeeze(np.array([x_min, x_min, x_min]))
        if len(x_max) == 1:
            x_max = np.squeeze(np.array([x_max, x_max, x_max]))

        shape_3D = np.array(densities[:, :, :, 0].shape)
        deltas = (x_max - x_min) / (shape_3D - 1)
        corners = x_min

    corners = corners.astype(np.float64)
    deltas = deltas.astype(np.float64)

    assert densities.flags.c_contiguous
    assert vectors.flags.c_contiguous
    assert insert_vals.flags.c_contiguous
    assert corners.flags.c_contiguous
    assert deltas.flags.c_contiguous

    fortran.density_f.trilinear_insertions_real(densities.T, vectors.T, insert_vals.T, corners.T, deltas.T)
    return None


def trilinear_insert(data_coord, data_val, x_min, x_max, n_bin, mask, boundary_mode="truncate", verbose=True):
    r"""
    Trilinear insertion on a regular grid with arbitrarily positioned sample points.

    This function returns two arrays, dataout and weightout.
    weightout is a 3D array containing the accumulated trilinear weights.
    dataout is the accumulated trilinearly inserted values.
    One needs to divide dataout by weightout (taking care to deal with zeros in weightout) to get the
    correct trilinear insertion result.
    This is so that the function can be used to sum over many trilinearly inserted arrays in
    for example a 3D diffracted intensity merge.

    Note 1: All input arrays should be C contiguous.
    Note 2: This code will break if you put a 1 in any of the N_bin entries.
    Note 3: The boundary is defined as [x_min-0.5, x_max+0.5).

    Arguments:
        data_coord (Nx3 |ndarray|) : Coordinates (x,y,z) of the data points that you wish to insert into
                     the regular grid.
        data_val (Nx1 |ndarray|) : The values of the data points to be inserted into the grid.
        x_min (1x3 |ndarray|) : (x_min, y_min, z_min)
        x_max (1x3 |ndarray|) : (x_max, y_max, z_max)
        n_bin (1x3 |ndarray|) : Number of bins in each direction (N_x, N_y, N_z)
        mask (Nx1 |ndarray|) : Specify which data points to ignore. Non-zero means use, zero means ignore.
        boundary_mode (str) : Specify how the boundary should be treated. Options are:
                              (1) "truncate" - Ignores all points outside the insertion volume.
                              (2) "periodic" - Equivalent to wrapping around.

    Returns:
        2-element tuple containing the following

        - **dataout** (*3D |ndarray|*) : Trilinearly summed values that needs to be divided by weightout to give the
          trilinearly inserted values.
        - **weightout** (*3D |ndarray|*) : Cumulative trilinear weights.
    """

    # Checks
    if fortran is None:
        raise ImportError('You need to compile fortran code to use utils.trilinear_interpolation()')
    if len(data_coord) != len(data_val):
        raise ValueError('The data coordinates and data values must be of the same length.')
    if len(data_coord) != len(mask):
        raise ValueError('The data coordinates and data mask must be of the same length.')
    if len(x_min) != 3:
        raise ValueError('x_min needs to be an array that contains three elements.')
    if len(x_max) != 3:
        raise ValueError('x_max needs to be an array that contains three elements.')
    if len(n_bin) != 3:
        raise ValueError('N_bin needs to be an array that contains three elements.')
    if data_coord.shape[1] != 3:
        raise ValueError('data_coord needs to be an Nx3 array.')

    if np.sum(x_min <= np.min(data_coord, axis=0)) != 3:
        if verbose:
            print('Warning: Values in data_coord is less than one or more of the limits specified in x_min. \n' +
              'I.e., one or more points are outside the insertion volume. \n' +
              'If this is intended, please disregard this message. \n' +
              'Else consider doing the following: np.min(data_coord, axis=0) and compare against x_min to see.\n')
    if np.sum(x_max >= np.max(data_coord, axis=0)) != 3:
        if verbose:
            print('Warning: Values in data_coord is greater than one or more of the limits specified in x_max. \n' +
              'I.e., one or more points are outside the insertion volume. \n' +
              'If this is intended, please disregard this message. \n' +
              'Else consider doing the following: np.min(data_coord, axis=0) and compare against x_min to see.\n')

    # Check if the non-1D arrays are c_contiguous
    assert data_coord.flags.c_contiguous

    # Store the datatype of the incoming data
    data_val_type = data_val.dtype

    # Convert to appropriate types for the insertion
    data_coord = data_coord.astype(np.double)
    data_val = data_val.astype(np.complex128)
    x_min = x_min.astype(np.double)
    x_max = x_max.astype(np.double)
    n_bin = n_bin.astype(np.int64)

    # Bin width
    delta_x = (x_max - x_min) / (n_bin - 1)

    # Bin volume
    bin_volume = delta_x[0] * delta_x[1] * delta_x[2]
    one_over_bin_volume = 1 / bin_volume

    # To safeguard against round-off errors
    epsilon = 1e-9

    # Constants (these are arrays with 3 elements in them)
    c1 = 0.0 - x_min / delta_x
    c2 = x_max + 0.5 - epsilon
    c3 = x_min - 0.5 + epsilon

    if boundary_mode == 'truncate':
        # Modify the mask to mask out points outside the insertion volume.

        # All three coordinates of a point needs to evaluate to true for the point to be
        # included in the insertion volume.
        mask_out_of_bound_coords_min = np.sum((x_min - delta_x) <= data_coord, axis=1) == 3
        mask_out_of_bound_coords_max = np.sum((x_max + delta_x) >= data_coord, axis=1) == 3

        # Update mask
        mask *= mask_out_of_bound_coords_min * mask_out_of_bound_coords_max

        # Mask out data_coord and data_val - user input mask
        data_coord = data_coord[mask != 0, :]
        data_val = data_val[mask != 0]

        # Initialise memory for Fortran
        # The N_bin+2 is for boundary padding when doing the interpolation
        data_out = np.zeros(n_bin + 2, dtype=np.complex128, order='C')
        weightout = np.zeros(n_bin + 2, dtype=np.double, order='C')
        data_out = np.asfortranarray(data_out)
        weightout = np.asfortranarray(weightout)

        # Mask out data_coord and data_val - user input x_min and x_max, i.e. mask out out-of-bounds data points
        # If any coordinate is greater than the maximum range (c2), throw it away.
        ind_outofbound_max = np.sum((data_coord - c2) > 0, axis=1) == 0
        data_coord = data_coord[ind_outofbound_max]
        data_val = data_val[ind_outofbound_max]

        # If any coordinate is less than the minimum range (c3), throw it away
        ind_outofbound_min = np.sum((data_coord - c3) < 0, axis=1) == 0
        data_coord = data_coord[ind_outofbound_min]
        data_val = data_val[ind_outofbound_min]

        # Number of data points - very crucial that this line is placed here
        # because N_data can change depending on if any sample points are out of
        # bounds.
        n_data = len(data_val)

        # Do trilinear insertion
        fortran.interpolations_f.trilinear_insert(data_coord, data_val, x_min, n_data,
                                                  delta_x, one_over_bin_volume, c1,
                                                  data_out, weightout)

        # Keep only the inner array - get rid of the boundary padding.
        data_out = data_out[1:n_bin[0] + 1, 1:n_bin[1] + 1, 1:n_bin[2] + 1]
        weightout = weightout[1:n_bin[0] + 1, 1:n_bin[1] + 1, 1:n_bin[2] + 1]

    elif boundary_mode == 'periodic':
        # Periodic boundary conditions on the insertion volume.

        # Mask out data_coord and data_val - user input mask
        data_coord = data_coord[mask != 0, :]
        data_val = data_val[mask != 0]

        # Initialise memory for Fortran
        data_out = np.zeros(n_bin, dtype=np.complex128, order='C')
        weightout = np.zeros(n_bin, dtype=np.double, order='C')
        data_out = np.asfortranarray(data_out)
        weightout = np.asfortranarray(weightout)

        # Number of data points
        n_data = len(data_val)

        # Do trilinear insertion
        fortran.interpolations_f.trilinear_insert_with_wraparound(data_coord, data_val, x_min, n_data,
                                                                  delta_x, one_over_bin_volume, c1, n_bin,
                                                                  data_out, weightout)
    else:
        raise ValueError('Unrecognized boundary mode')

    # If the original datatype is not complex, then return only the real part.
    if data_val_type != np.complex128:
        data_out = np.real(data_out)

    return data_out, weightout

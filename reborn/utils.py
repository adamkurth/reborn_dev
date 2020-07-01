r"""
Some utility functions that might be useful throughout reborn.  Don't put highly specialized functions here.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from functools import wraps
import sys
import os
import pkg_resources
import numpy as np
from numpy import sin, cos
from numpy.fft import fftshift, fft, ifft
# from numba import jit
from . import fortran


def docs():
    r""" Open the reborn documentation in a web browser (if available)."""

    docs_file_path = pkg_resources.resource_filename('reborn', '')
    docs_file_path = os.path.join(docs_file_path, '..', 'doc', 'html', 'index.html')

    if os.path.exists(docs_file_path):
        docs_file_path = 'file://' + docs_file_path
    else:
        docs_file_path = 'https://rkirian.gitlab.io/reborn'

    try:
        import webbrowser
    except ImportError:
        print("Can't open docs because you need to install the webbrowser Python package.")
        print("If using conda, perhaps you could run 'conda install webbrowser'")
        print('You can otherwise point your webbrowser to https://rkirian.gitlab.io/reborn')
        return

    webbrowser.open('file://' + docs_file_path)


def ensure_list(obj):
    r"""
    Make sure that some object is a list.  This is helpful because, for example, we frequently write code around the 
    assumption that detector geometry comes in the form of a list of |PADGeometry| instances.  However, it is also not
    so uncommon to have a single |PADGeometry|.  
    
    This function is very simple:
    
    .. code-block:: python
    
        def ensure_list(obj):
            if isinstance(obj, list):
                return obj
            if isinstance(obj, tuple):
                return list(obj)
            return [obj]
    
    Arguments:
        obj (object): The object that we want to ensure is a list.

    Returns: list
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return [obj]


def vec_norm(vec):
    r"""
    Compute normal vectors, which have lengths of one.

    Arguments:
        vec (numpy array of shape Nx3): Input vector

    Returns:
        numpy array of shape Nx3 : New unit vectors
    """

    vecnorm = np.sqrt(np.sum(vec**2, axis=(vec.ndim - 1)))
    return (vec.T / vecnorm).T


def vec_mag(vec):
    r"""
    Compute the scalar magnitude of an array of vectors.

    Arguments:
        vec (numpy array): Input array of vectors, shape (N, 3)

    Returns: scalar vector magnitudes
    """

    return np.sqrt(np.sum(vec * vec, axis=(vec.ndim - 1)))


def depreciate(message):
    r"""
    Utility for sending warnings when some class, method, function, etc. is depreciated.  It simply prints a message of
    the form "WARNING: DEPRECIATION: blah blah blah" but perhaps it will do someting more sophisticated in the future
    if the need arises.

    Arguments:
        message: whatever you want to have printed to the screen

    Returns: None
    """
    warn('DEPRECIATION: ' + message)


def warn(message):
    r"""
    Standard way of sending a warning message.  As of now this simply results in a function call

    sys.stdout.write("WARNING: %s\n" % message)

    The purpose of this function is that folks can search for "WARNING:" to find all warning messages, e.g. with grep.

    Arguments:
        message: the message you want to have printed.

    Returns: None
    """

    sys.stdout.write("WARNING: %s\n" % message)


def error(message):
    r"""
    Standard way of sending an error message.  As of now this simply results in a function call

    sys.stdout.write("ERROR: %s\n" % message)

    Arguments:
        message: the message you want to have printed.

    Returns: None
    """

    sys.stderr.write("ERROR: %s\n" % message)


def random_rotation():
    r"""
    This function has been removed.  Use scipy instead:

    .. code-block:: python

        from scipy.spatial.transform import Rotation
        rotmat = Rotation.random().as_matrix()'
    """

    depreciate('reborn.utils.random_rotation has been removed.  Use scipy for this:\n'
               'from scipy.spatial.transform import Rotation\n'
               'rotmat = Rotation.random().as_matrix()')

    return None


def rotation_about_axis(theta, vec):
    r"""
    This needs to be tested.  It was taken from
    https://stackoverflow.com/questions/17763655/rotation-of-a-point-in-3d-about-an-arbitrary-axis-using-python

    Arguments:
        theta (float): Rotation angle
        vec (numpy array): 3D vector specifying rotation axis

    Returns (numpy array): The shape (3, 3) rotation matrix
    """

    vec = vec_norm(np.array(vec)).reshape(3)
    ct = cos(theta)
    st = sin(theta)
    rot = np.array([[ct + vec[0] ** 2 * (1 - ct),
                     vec[0] * vec[1] * (1 - ct) - vec[2] * st,
                     vec[0] * vec[2] * (1 - ct) + vec[1] * st],
                    [vec[0] * vec[1] * (1 - ct) + vec[2] * st,
                     ct + vec[1] ** 2 * (1 - ct),
                     vec[1] * vec[2] * (1 - ct) - vec[0] * st],
                    [vec[0] * vec[2] * (1 - ct) - vec[1] * st,
                     vec[1] * vec[2] * (1 - ct) + vec[0] * st,
                     ct + vec[2] ** 2 * (1 - ct)]])
    return rot.reshape(3, 3)


def random_unit_vector():
    r"""
    Generate a totally random unit vector.

    Returns: numpy array length 3

    """
    return vec_norm(np.random.normal(size=3))


def random_beam_vector(div_fwhm):
    r"""
    A random vector for emulating beam divergence. Generates a random normal vector that is nominally along the [0,0,1]
    direction but with a random rotation along the [1,0,0] axis with given FWHM (Gaussian distributed and centered about
    zero) followed by a random rotation about the [0,0,1] axis with uniform distribution in the interval [0,2*pi).

    Arguments:
        div_fwhm (float):  FWHM of divergence angle.  Assuming Gaussian, where sigma = FWHM / 2.3548

    Returns:
        (numpy array) of length 3
    """

    # Don't do anything if no divergence
    bvec = np.array([0, 0, 1.0])
    if div_fwhm == 0:
        return bvec

    # First rotate around the x axis with Gaussian prob. dist.
    sig = div_fwhm / 2.354820045
    theta = np.random.normal(0, sig, [1])[0]
    rtheta = rotation_about_axis(theta, [1.0, 0, 0])
    bvec = np.dot(rtheta, bvec)

    # Next rotate around z axis with uniform dist [0,2*pi)
    phi = np.random.random(1)[0] * 2 * np.pi
    rphi = rotation_about_axis(phi, [0, 0, 1.0])
    bvec = np.dot(rphi, bvec)
    bvec /= np.sqrt(np.sum(bvec**2))

    return bvec


# def random_mosaic_rotation(mosaicity_fwhm):
#     r"""
#     Attempt to generate a random orientation for a crystal mosaic domain.  This is a hack.
#     We take the matrix product of three rotations, each of the same FWHM, about the three
#     orthogonal axis.  The order of this product is a random permutation.
#
#     :param mosaicity_fwhm:
#     :return:
#     """
#
#     if mosaicity_fwhm == 0:
#         return np.eye(3)
#
#     rs = list()
#     rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [1.0, 0, 0]))
#     rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [0, 1.0, 0]))
#     rs.append(rotation_about_axis(np.random.normal(0, mosaicity_fwhm / 2.354820045, [1])[0], [0, 0, 1.0]))
#     rind = np.random.permutation([0, 1, 2])
#     return rs[rind[0]].dot(rs[rind[1]].dot(rs[rind[2]]))


def triangle_solid_angle(r1, r2, r3):
    r"""
    Compute solid angle of a triangle whose vertices are r1,r2,r3, using the method of
    Van Oosterom, A. & Strackee, J. Biomed. Eng., IEEE Transactions on BME-30, 125-126 (1983).

    Arguments:
        r1 (numpy array): Vectors to vertices 1; array of shape (N, 3)
        r2 (numpy array): Vectors to vertices 1; array of shape (N, 3)
        r3 (numpy array): Vectors to vertices 1; array of shape (N, 3)

    Returns:
        (numpy array) of length N with solid angles
    """

    numer = np.abs(np.sum(r1 * np.cross(r2, r3), axis=-1))

    r1_n = np.linalg.norm(r1, axis=-1)
    r2_n = np.linalg.norm(r2, axis=-1)
    r3_n = np.linalg.norm(r3, axis=-1)
    denom = r1_n * r2_n * r2_n
    denom += np.sum(r1 * r2, axis=-1) * r3_n
    denom += np.sum(r2 * r3, axis=-1) * r1_n
    denom += np.sum(r3 * r1, axis=-1) * r2_n
    s_ang = np.arctan2(numer, denom) * 2

    return s_ang


def __fake_numba_jit(*args, **kwargs):
    r"""
    This is a fake decorator.  It is presently used to avoid errors when numba is missing, but will usually result in
    very slow code.

    Note: do not use this function.  It will go away some day.
    """
    def decorator(func):
        print('You need to install numba, else your code will run very slowly.')
        return func
    return decorator


def memoize(function):
    r"""
    This is a function decorator for caching results from a function, to avoid
    excessive computation or reading from disk.  Search the web for more
    details of how this works.
    """

    memo = {}

    @wraps(function)
    def wrapper(*args):

        if args in memo:
            return memo[args]

        rv = function(*args)
        memo[args] = rv
        return rv

    return wrapper


def max_pair_distance(vecs):
    r"""
    Determine the maximum distance between to vectors in a list of vectors.

    Arguments:
        vecs (Nx3 numpy array) : Input vectors.

    Returns:
        float : The maximum pair distance.
    """
    vecs = np.double(vecs)
    if not vecs.flags.c_contiguous:
        vecs = vecs.copy()
    d_max = np.array([0], dtype=np.float64)
    fortran.utils_f.max_pair_distance(vecs.T, d_max)
    return d_max[0]


def trilinear_insert(data_coord, data_val, x_min, x_max, n_bin, mask, boundary_mode="truncate"):
    r"""
    Trilinear insertion on a regular grid with arbitrary sample points.
    The boundary is defined as [x_min-0.5, x_max+0.5).
    This function returns two arrays, dataout and weightout, you need to divide dataout
    by weightout (taking care to deal with zeros in weightout) to get the correct trilinear
    insertion result. This is so that the function can be used to sum over many trilinearly
    inserted arrays in for example a 3D diffracted intensity merge.

    Note 1: All input arrays should be C contiguous.
    Note 2: This code will break if you put a 1 in any of the N_bin entries.

    Arguments:
        data_coord (Nx3 numpy array) : Coordinates (x,y,z) of the data points that you wish to insert into
                     the regular grid.
        data_val (Nx1 numpy array) : The values of the data points to be inserted into the grid.
        x_min (1x3 numpy array) : (x_min, y_min, z_min)
        x_max (1x3 numpy array) : (x_max, y_max, z_max)
        n_bin (1x3 numpy array) : Number of bins in each direction (N_x, N_y, N_z)
        mask (Nx1 numpy array) : Specify which data points to ignore. Non-zero means use, zero means ignore.
        boundary_mode (str) : Specify how the boundary should be treated. Options are:
                              (1) "truncate" - Ignores all points outside the insertion volume.
                              (2) "periodic" - Equivalent to wrapping around.

    Returns:
        2-element tuple containing the following

        - **dataout** (*3D numpy array*) : Trilinearly summed values that needs to be divided by weightout to give the
          trilinearly inserted values.
        - **weightout** (*3D numpy array*) : Number of times each voxel has a value put into it.
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
        print('Warning: Values in data_coord is less than one or more of the limits specified in x_min. \n' +
              'I.e., one or more points are outside the insertion volume. \n' +
              'If this is intended, please disregard this message. \n' +
              'Else you could execute the following code: np.min(data_coord, axis=0) and compare against x_min to see. \n')
    if np.sum(x_max >= np.max(data_coord, axis=0)) != 3:
        print('Warning: Values in data_coord is greater than one or more of the limits specified in x_max. \n' + 
              'I.e., one or more points are outside the insertion volume. \n' + 
              'If this is intended, please disregard this message. \n' +
              'Else you cold execute the following code: np.max(data_coord, axis=0) and compare against x_max to see. \n')

    # Check if the non-1D arrays are c_contiguous
    assert data_coord.flags.c_contiguous

    # Convert to appropriate types
    data_coord = data_coord.astype(np.double)
    data_val = data_val.astype(np.complex128)
    x_min = x_min.astype(np.double)
    x_max = x_max.astype(np.double)
    n_bin = n_bin.astype(np.int)

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
        dataout = np.zeros(n_bin + 2, dtype=np.complex128, order='C')
        weightout = np.zeros(n_bin + 2, dtype=np.double, order='C')
        dataout = np.asfortranarray(dataout)
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
        N_data = len(data_val)

        # Do trilinear insertion
        fortran.interpolations_f.trilinear_insert(data_coord, data_val, x_min, N_data,
                                                  delta_x, one_over_bin_volume, c1,
                                                  dataout, weightout)

        # Keep only the inner array - get rid of the boundary padding.
        dataout = dataout[1:n_bin[0] + 1, 1:n_bin[1] + 1, 1:n_bin[2] + 1]
        weightout = weightout[1:n_bin[0] + 1, 1:n_bin[1] + 1, 1:n_bin[2] + 1]

    elif boundary_mode == 'periodic':
        # Periodic boundary conditions on the insertion volume.

        # Mask out data_coord and data_val - user input mask
        data_coord = data_coord[mask != 0, :]
        data_val = data_val[mask != 0]

        # Initialise memory for Fortran
        dataout = np.zeros(n_bin, dtype=np.complex128, order='C')
        weightout = np.zeros(n_bin, dtype=np.double, order='C')
        dataout = np.asfortranarray(dataout)
        weightout = np.asfortranarray(weightout)

        # Number of data points
        N_data = len(data_val)

        # Do trilinear insertion
        fortran.interpolations_f.trilinear_insert_with_wraparound(data_coord, data_val, x_min, N_data,
                                                                  delta_x, one_over_bin_volume, c1, n_bin,
                                                                  dataout, weightout)

    # The code in this section is no longer used because we want to return both dataout and weightout.
    # Returning weightout is useful when we want to interpolate the interpolated slices,
    # for example in the case of merging slices for a 3D volume.

    # # Calculate the mean value inserted into the array by dividing dataout by weightout.
    # # For locations where weightout is zero, dataout should also be zero (because no values were inserted),
    # # deal with this case by setting weightout to 1.
    # assert np.sum(dataout[weightout == 0]) == 0

    return dataout, weightout


def rotate3D(f, Euler_angles):
    r"""
    Rotate a 3D array of numbers in 3-dimensions.
    The function works by rotating each 2D sections of the 3D array via three shears,
    as described by Unser et al. (1995) "Convolution-based interpolation for fast, 
    high-quality rotation of images." IEEE Transactions on Image Processing, 4:1371.

    Note 1: If the input array, f, is non-cubic, it will be zero-padded to a cubic array
            with length the size of the largest side of the original array.

    Note 2: If you don't want wrap arounds, make sure the input array, f, is zero-padded to
            at least sqrt(2) times the largest dimension of the desired object.

    Note 3: Proper Euler angle convention is used, i.e, zyz.

    Arguments:
        f (*3D numpy array*) : The 3D input array.
        Euler_angles (1x3 numpy array) : The three Euler angles, in zyz format.

    Returns:
        - **f_rot** (*3D numpy array*) : The rotated 3D array.
    """

    #---------------------------
    # Define private functions

    def rotate90(f):
       return np.transpose(np.fliplr(f))


    def rotate180(f):
       return np.fliplr(np.flipud(f))


    def rotate270(f):
       return np.transpose(np.flipud(f))


    def shiftx(f, kxfac, xfac):
        return ifft(fftshift(fft(f, axis=0), axes=0) * kxfac, axis=0) * xfac


    def shifty(f, kyfac, yfac):
        return ifft(fftshift(fft(f, axis=1), axes=1) * kyfac, axis=1) * yfac


    def rotate2D(fr, kxfac, xfac, kyfac, yfac, n90_mod_Four):
        """ Rotate a 2D section. """
        
        if (n90_mod_Four == 1):
            fr = rotate90(fr)
        elif (n90_mod_Four == 2):
            fr = rotate180(fr)
        elif (n90_mod_Four == 3):
            fr = rotate270(fr)

        fr = shiftx(fr, kxfac, xfac)
        fr = shifty(fr, kyfac, yfac)
        fr = shiftx(fr, kxfac, xfac)

        return fr

    def rotate_setup(f, ang):
        """ Set up required to rotate. """
        n90 = np.rint(ang*TwoOverPi)
        dang = ang - n90 * PiOvTwo

        t = -np.tan(0.5*dang)
        s = np.sin(dang)

        kxfac = np.exp(constx1 * t)
        xfac = np.exp(constx2_2 - constx2_3 * t)

        kyfac = np.exp(consty1 * s)
        yfac = np.exp(consty2_2 - consty2_3 * s)

        n90_mod_Four = n90 % 4

        return kxfac, xfac, kyfac, yfac, n90_mod_Four


    def __rotate_Euler_z(f, ang):

        kxfac, xfac, kyfac, yfac, n90_mod_Four = rotate_setup(f, ang)
        
        f_rot = np.zeros((N, N, N), dtype=np.complex128)
        for ii in range(0, N):
            f_rot[ii,:,:] = rotate2D(f[ii,:,:], kxfac, xfac, kyfac, yfac, n90_mod_Four)

        return f_rot


    def __rotate_Euler_y(f, ang):

        kxfac, xfac, kyfac, yfac, n90_mod_Four = rotate_setup(f, ang)

        f_rot = np.zeros((N, N, N), dtype=np.complex128)
        for ii in range(0, N):
            f_rot[:,ii,:] = rotate2D(f[:,ii,:], kxfac, xfac, kyfac, yfac, n90_mod_Four)

        return f_rot
    #---------------------------
    # Get the max shape of the array
    Nx, Ny, Nz = f.shape
    N = np.max([Nx, Ny, Nz])

    # Make array cubic if the array is not cubic.
    if Nx != Ny or Nx != Nz or Ny != Nz:
        f_rot = np.zeros((N,N,N), dtype=np.complex128)
        f_rot[0:Nx, 0:Ny, 0:Nz] = f
    else:
        f_rot = f

    #---------------------------
    # Precalculations for speed
    Y, X = np.meshgrid(np.arange(N), np.arange(N))

    y0 = 0.5 * (N-1)
    constx1   = -1j*2.0*np.pi/N * X * (Y-y0)
    constx2_1 = -1j*np.pi * (1-(N%2)/N)
    constx2_2 = constx2_1*X
    constx2_3 = constx2_1*(Y-y0)

    x0 = 0.5 * (N-1)
    consty1   = -1j*2.0*np.pi/N * Y * (X-x0)
    consty2_1 = -1j*np.pi * (1-(N%2)/N)
    consty2_2 = consty2_1*Y
    consty2_3 = consty2_1*(X-x0)

    TwoOverPi = 2.0/np.pi
    PiOvTwo = 0.5 * np.pi

    #---------------------------
    # Do the rotations
    f_rot = __rotate_Euler_z(f_rot, ang=Euler_angles[0])
    f_rot = __rotate_Euler_y(f_rot, ang=Euler_angles[1])
    f_rot = __rotate_Euler_z(f_rot, ang=Euler_angles[2])

    return f_rot



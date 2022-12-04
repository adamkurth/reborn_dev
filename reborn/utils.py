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
r"""
Some utility functions that might be useful throughout reborn.  Don't put highly specialized functions here.
"""
from functools import wraps
import os
import inspect
import logging
import hashlib
import subprocess
import pkg_resources
import numpy as np
from numpy import sin, cos
from numpy.fft import fftshift, fftn
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from .config import configs

logger = logging.getLogger()


def docs():
    r""" Open the reborn documentation in a web browser (if available)."""
    docs_file_path = pkg_resources.resource_filename('reborn', '')
    docs_file_path = os.path.join(docs_file_path, '..', 'doc', 'html', 'index.html')
    if os.path.exists(docs_file_path):
        docs_file_path = 'file://' + docs_file_path
    else:
        docs_file_path = 'https://rkirian.gitlab.io/reborn'
    try:
        import webbrowser  # pylint: disable=import-outside-toplevel
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

    This function does the following simple task:

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
        vec (|ndarray|): Input vectors.  Array shape: (N, 3).

    Returns:
        (|ndarray|): New unit vectors.  Array shape: (N, 3).
    """
    vecnorm = np.sqrt(np.sum(vec ** 2, axis=(vec.ndim - 1)))
    return (vec.T / vecnorm).T


def vec_mag(vec):
    r"""
    Compute the scalar magnitude of an array of vectors.

    Arguments:
        vec (|ndarray|): Input array of vectors, shape (N, 3)

    Returns:
        |ndarray|: Scalar vector magnitudes
    """
    return np.sqrt(np.sum(vec * vec, axis=(vec.ndim - 1)))


def vec_shape(vec):
    r"""
    Ensure that an array has the proper shape that is expected of 3-vector arrays.  They should always be 2-dimensional,
    with shape (N, 3).

    Arguments:
        vec (|ndarray|): The vector array.

    Returns:
        |ndarray| with shape (N, 3)

    """
    vec = atleast_2d(np.squeeze(vec))
    if len(vec.shape) != 2:
        raise ValueError('Something is wrong with your vector array shape. It should be (N, 3)')
    if vec.shape[1] != 3:
        raise ValueError('Something is wrong with your vector array shape. It should be (N, 3).')
    return vec


def depreciate(*args, caller=0, **kwargs):
    r"""
    Standard way of printing a depreciation message.  It is presently just a wrapper around the print function,
    but prepends "WARNING:DEPRECIATION:<caller>:" to the error message, where <caller> is (optionally) the name of the
    function from which the message originated.

    Arguments:
        args: Standard print function positional arguments
        kwargs: Standard print function keyword arguments
        caller (int): To include the caller in the message, set to an integer >= 1.

    Returns: None
    """
    msg = "*** WARNING *** DEPRECIATION:"
    if caller >= 0:
        msg += get_caller(1)
    msg += ':'
    print(msg, *args, **kwargs)


def warn(*args, caller=0, **kwargs):
    r"""
    Standard way of printing a warning message.  It is presently just a wrapper around the print function, but prepends
    "WARNING:<caller>:" to the error message, where <caller> is (optionally) the name of the function from which the
    message originated.

    Arguments:
        args: Standard print function positional arguments
        kwargs: Standard print function keyword arguments
        caller (int): To include the caller in the message, set to an integer >= 1.

    Returns: None
    """
    msg = "WARNING:"
    if caller >= 0:
        msg += get_caller(1)
    msg += ':'
    print(msg, *args, **kwargs)


def debug(*args, caller=0, **kwargs):
    r"""
    Standard way of printing a debug message.  It is presently just a wrapper around the print function, but prepends
    "DEBUG:<caller>:" to the error message, where <caller> is (optionally) the name of the function from which the
    message originated.

    Arguments:
        args: Standard print function positional arguments
        kwargs: Standard print function keyword arguments
        caller (int): To include the caller in the message, set to an integer >= 1.

    Returns: None
    """
    if not configs['debug']:
        return
    msg = "DEBUG:"
    if caller >= 0:
        msg += get_caller(1)
    msg += ':'
    print(msg, *args, **kwargs)


def error(*args, caller=0, **kwargs):
    r"""
    Standard way of printing an error message.  It is presently just a wrapper around the print function, but prepends
    "ERROR:<caller>:" to the error message, where <caller> is (optionally) the name of the function from which the
    message originated.

    Arguments:
        args: Standard print function positional arguments
        kwargs: Standard print function keyword arguments
        caller (int): To include the caller in the message, set to an integer >= 1.

    Returns: None
    """
    msg = "ERROR:"
    if caller >= 0:
        msg += get_caller(1)
    msg += ':'
    print(msg, *args, **kwargs)


def random_rotation():
    r"""
    Create a random rotation matrix.

    .. code-block:: python

        from scipy.spatial.transform import Rotation
        rotmat = Rotation.random().as_matrix()'
    """
    return Rotation.random().as_matrix()


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

    Returns:
        |ndarray|
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
    bvec /= np.sqrt(np.sum(bvec ** 2))

    return bvec


def triangle_solid_angle(r1, r2, r3):
    r"""
    Compute solid angle of a triangle whose vertices are r1,r2,r3, using the method of
    Van Oosterom, A. & Strackee, J. Biomed. Eng., IEEE Transactions on BME-30, 125-126 (1983).

    Arguments:
        r1 (|ndarray|): Vectors to vertices 1; array of shape (N, 3)
        r2 (|ndarray|): Vectors to vertices 1; array of shape (N, 3)
        r3 (|ndarray|): Vectors to vertices 1; array of shape (N, 3)

    Returns:
        (|ndarray|) of length N with solid angles
    """

    top = np.abs(np.sum(r1 * np.cross(r2, r3), axis=-1))

    r1_n = np.linalg.norm(r1, axis=-1)
    r2_n = np.linalg.norm(r2, axis=-1)
    r3_n = np.linalg.norm(r3, axis=-1)
    bottom = r1_n * r2_n * r2_n
    bottom += np.sum(r1 * r2, axis=-1) * r3_n
    bottom += np.sum(r2 * r3, axis=-1) * r1_n
    bottom += np.sum(r3 * r1, axis=-1) * r2_n
    s_ang = np.arctan2(top, bottom) * 2

    return s_ang


def __fake_numba_jit(*args, **kwargs):  # pylint: disable=unused-argument
    r"""
    This is a fake decorator.  It is presently used to avoid errors when numba is missing, but will usually result in
    very slow code.

    Note: do not use this function.  It will go away some day.

    FIXME: Get rid of everything that uses numba.  Use fortran instead.
    """

    def decorator(func):
        r"""FIXME: Docstring."""
        print('You need to install numba, else your code will run very slowly.')
        return func

    return decorator


def memoize(function):
    r"""
    This is a function decorator for caching results from a function.  It is used, for example, to avoid re-loading
    data files containing scattering factors.  We assume that your computer has enough RAM to handle this, and we assume
    that the developers will not abuse this feature.
    FIXME: This needs to be tested.  It was blindly copied from the internet...
    FIXME: Consider adding a configuration to reborn that disallows the use of memoize.
    """
    memo = {}

    @wraps(function)
    def wrapper(*args):
        r""" FIXME: Docstring."""
        if args in memo:
            return memo[args]
        rv = function(*args)
        memo[args] = rv
        return rv

    return wrapper


def max_pair_distance(*args, **kwargs):
    r"""
    Depreciated.  Use reborn.utils.vectors.max_pair_distance
    """
    depreciate('Do not use reborn.utils.max_pair_distance.  Use reborn.misc.vectors.max_pair_distance.')
    from .misc.vectors import max_pair_distance
    return max_pair_distance(*args, **kwargs)


def make_label_radial_shell(r_bin_vec, n_vec):
    r"""
    For fast radial statistics calculations - done through a precomputed label array.

    Produce a 3D volume with concentric shells of thickness specified by r_bin_vec.
    Each shell is labelled incrementally by integers starting with 1 at the centre.
    (the label zero is reserved for masked values in the volume).

    Voxels outside the maximum radius is given a label of zero.

    Arguments:
        r_bin_vec - Radii of the shells - in voxel units
        n_vec     - Shape of the desired volume

    Returns:
        labels_radial
    """
    nx, ny, nz = n_vec

    nx_cent = int(nx / 2)  # equivalent to int(np.floor(nx/2))
    ny_cent = int(ny / 2)
    nz_cent = int(nz / 2)

    # Initialise memory
    labels_radial = np.zeros((nx, ny, nz), dtype=np.int)

    r_bin_vec_sq = r_bin_vec ** 2

    for i in range(nx):
        i_sq = (i - nx_cent) ** 2

        for j in range(ny):
            j_sq = (j - ny_cent) ** 2

            for k in range(nz):
                r_sq = i_sq + j_sq + (k - nz_cent) ** 2

                # Get the index of which r bin the current (i,j,k) voxel belongs
                r_ind = np.sum(r_sq >= r_bin_vec_sq)

                labels_radial[i, j, k] = r_ind

    # This is to make the centre voxel exactly have its unique label of 1.
    labels_radial += 1  # Shift all labels up by 1.
    labels_radial[nx_cent, ny_cent, nz_cent] = 1  # Now make the centre 1.

    # Set all voxels lying outside a radius larger than the final value in r_bin_vec to a label of zero.
    labels_radial[labels_radial == len(r_bin_vec) + 1] = 0

    return labels_radial


def radial_stats(f, labels_radial, n_radials, mode):
    r"""
    Calculate the statistics of the voxels in each shell.

    Input:
        f              - The input 3D array of numbers
        labels_radial  - The labels
        n_radials      - Maximum label value
        mode           - The desired statistics that we wish to calculate

    Output:
        radial_stats_vec
    """

    # Initialise memory
    f_dtype = f.dtype
    if f_dtype == np.float64:
        radial_stats_vec = np.zeros(n_radials, dtype=np.float64)
    elif f_dtype == np.complex128:
        radial_stats_vec = np.zeros(n_radials, dtype=np.complex128)
    else:
        raise ValueError("Data type not implemented.")

    # Calculate the radial stats
    # Range runs from 1 to N_radials+1 to ignore the zero label
    if mode == "mean":
        for i in range(n_radials):
            radial_stats_vec[i] = np.mean(f[labels_radial == i + 1])
    elif mode == "sum":
        for i in range(n_radials):
            radial_stats_vec[i] = np.sum(f[labels_radial == i + 1])
    elif mode == "count":  # Number of voxels in each shell
        for i in range(n_radials):
            radial_stats_vec[i] = np.sum(labels_radial == i + 1)
    elif mode == "median":
        for i in range(n_radials):
            radial_stats_vec[i] = np.median(f[labels_radial == i + 1])
    else:
        raise ValueError("Mode not recognised.")

    return radial_stats_vec


def get_FSC(f1, f2, labels_radial, n_radials):
    r"""
    Calculate the Fourier shell correlation (FSC) between two 3D numpy arrays.
    """
    F1 = fftshift(fftn(f1))
    F2 = fftshift(fftn(f2))

    radial_f1_f2 = radial_stats(f=F1 * np.conj(F2), labels_radial=labels_radial, n_radials=n_radials, mode="sum")
    radial_f1 = radial_stats(f=(np.abs(F1) ** 2).astype(np.float64), labels_radial=labels_radial, n_radials=n_radials,
                             mode="sum")
    radial_f2 = radial_stats(f=(np.abs(F2) ** 2).astype(np.float64), labels_radial=labels_radial, n_radials=n_radials,
                             mode="sum")

    return np.abs(radial_f1_f2) / np.abs((np.sqrt(radial_f1) * np.sqrt(radial_f2)))


def atleast_1d(x):
    r""" Expand dimensions of |ndarray|.  Add dimensions to the left-most index. """
    x = np.array(x)
    if x.ndim < 1:
        x = np.expand_dims(x, axis=0)
    return x


def atleast_2d(x):
    r""" Expand dimensions of |ndarray|.  Add dimensions to the left-most index. """
    x = np.array(x)
    if x.ndim < 2:
        x = np.expand_dims(x, axis=0)
    return x


def atleast_3d(x):
    r""" Expand dimensions of |ndarray|.  Add dimensions to the left-most index. """
    x = np.array(x)
    if x.ndim < 3:
        x = np.expand_dims(atleast_2d(x), axis=0)
    return x


def atleast_4d(x):
    r""" Expand dimensions of |ndarray|.  Add dimensions to the left-most index. """
    x = np.array(x)
    if x.ndim < 4:
        x = np.expand_dims(atleast_3d(x), axis=0)
    return x


def binned_statistic(x, y, func, n_bins, bin_edges, fill_value=0):
    r""" Similar to :func:`binned_statistic <scipy.stats.binned_statistic>` but faster because regular bin spacing
    is assumed, and sparse matrix algorithms are used.  Speedups of ~30-fold have been observed.

    Based on the discussion found here:
    https://stackoverflow.com/questions/26783719/efficiently-get-indices-of-histogram-bins-in-python

    Args:
        x (|ndarray|): The coordinates that correspond to values y below.  Bin indices derive from x.
        y (|ndarray|): The values that enter into the statistic.
        func (function): The function that will be applied to values in each bin
        n_bins (int): Desired number of bins
        bin_edges (tuple of floats): The min and max edges of bins (edges, not the centers, of the bins)
        fill_value (float): Initialize output array with these values.

    Returns:
        |ndarray|
    """
    n_data = len(y)
    r0, r1 = bin_edges
    bin_size = float(r1 - r0) / n_bins
    # To avoid out-of-range values, we add a bin to the left and right.  Out-of-range entries get tossed into those
    # bins, which we throw away later.
    n_bins += 2
    r0 -= bin_size
    r1 += bin_size
    digitized = (float(n_bins) / (r1 - r0) * (x - r0)).astype(int)
    digitized = np.maximum(digitized, 0)
    digitized = np.minimum(digitized, n_bins - 1)
    mat = csr_matrix((y, [digitized, np.arange(n_data)]), shape=(n_bins, n_data))
    groups = np.split(mat.data, mat.indptr[1:-1])
    out = np.empty(n_bins, dtype=y.dtype)
    out.fill(fill_value)
    for i in range(n_bins):
        if len(groups[i]) > 0:
            out[i] = func(groups[i])
    return out[1:-1]


def get_caller(n=0):
    r""" Get the name of the function that calls this one. """
    stack = inspect.stack()
    if len(stack) > (n+1):
        return inspect.stack()[n+1][3]
    return 'get_caller'


def check_file_md5(file_path, md5_path=None):
    r"""
    Utility for checking if a file has been modified from a previous version.

    Given a file path, check for a file with ".md5" appended to the path.  If it exists, check if the md5 hash
    saved in the file matches the md5 hash of the current file and return True.  Otherwise, return False.

    Arguments:
        file_path (str): Path to the file.
        md5_path (str): Optional path to the md5 file.

    Returns:
        bool
    """
    if md5_path is None:
        md5_path = file_path+'.md5'
    if not os.path.exists(md5_path):
        return False
    with open(md5_path, 'r', encoding="utf-8") as f:
        md5 = f.readline()
    with open(file_path, 'rb') as f:
        hasher = hashlib.md5()
        hasher.update(f.read())
        new_md5 = str(hasher.hexdigest())
    debug(md5_path, 'md5', md5)
    debug(file_path, 'md5', new_md5)
    if new_md5 == md5:
        return True
    return False


def write_file_md5(file_path, md5_path=None):
    r"""
    Save the md5 hash of a file.  The output will be the same as the original file but with a '.md5' extension appended.

    Arguments:
        file_path (str): The path of the file to make an md5 from.
        md5_path (str): Optional path to the md5 file.

    Returns:
        str: the md5
    """
    if md5_path is None:
        md5_path = file_path+'.md5'
    with open(file_path, 'rb') as f:
        hasher = hashlib.md5()
        hasher.update(f.read())
        md5 = str(hasher.hexdigest())
    md5_prev = None
    if os.path.exists(md5_path):
        with open(md5_path, 'r', encoding="utf-8") as f:
            md5_prev = f.readline()
    if md5 != md5_prev:
        with open(md5_path, 'w', encoding="utf-8") as f:
            f.write(md5)
    return md5_path


def git_sha():
    r""" Return the SHA of the git repo is in the current directory.  Return None if the repo is not
    totally clean. Useful if you wish to keep track of the git version that produced your results.

    Returns: str
    """
    out = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if out.stderr == b'':
        if out.stdout == b'':
            sha = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, check=True).stdout
            return sha.decode("utf-8").strip()
    return None


def google_docstring_test():
    r"""
    This is an example docstring for developers.  This function does nothing.

    You might link to other reborn docs, such as :class:`reborn.detector.PADGeometry` .  Linking also works for some
    external packages, for example :func:`np.median`.

    There are various shortcuts in ``doc/conf.py``, such as |PADGeometry| and |Henke1993|.

    You can put an equation inline like this :math:`f(q)`, or on a separate line like this:

    .. math::
        a_i = \sum_n f_n \exp(-i \vec{q}_i \cdot (\mathbf{R} \vec{r} + \vec{U}))

    .. note::
        This is a note to emphasize something important.

    Unfortunately,

    Arguments:
        data (|ndarray| or list of |ndarray|): Data to get profiles from.
        beam (|Beam|): Beam info.
        pad_geometry (|PADGeometryList|): PAD geometry info.
        mask (|ndarray| or list of |ndarray|): Mask (one is good, zero is bad).
        n_bins (int): Number of radial bins.
        q_range (tuple of floats): Centers of the min and max q bins.
        statistic (function): The function you want to apply to each bin (default: :func:`np.mean`).

    Returns:
        (tuple):
            - **statistic** (|ndarray|) -- Radial statistic.
            - **bins** (|ndarray|) -- The values of q at the bin centers.
    """
    return None
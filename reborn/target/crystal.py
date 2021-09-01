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

# coding=utf-8
"""
Basic utilities for dealing with crystalline objects.
"""

import os
import tempfile
try:
    import urllib.request
except ImportError:
    import urllib
import pkg_resources
import numpy as np
from .molecule import Molecule
from . import atoms
from ..utils import warn, vec_mag, trilinear_insert
try:
    from numba import jit
except ImportError:
    from ..utils import __fake_numba_jit as jit


pdb_data_path = pkg_resources.resource_filename('reborn.data', 'pdb')
temp_dir = os.path.join(tempfile.gettempdir(), 'reborn')
os.makedirs(temp_dir, exist_ok=True)


def get_pdb_file(pdb_id, save_path=temp_dir, silent=False):
    r"""
    Download a PDB file from the PDB web server and return the path to the downloaded file.  There is a data directory
    included with reborn that includes a few PDB files for testing purposes - if the requested PDB file exists there,
    the path to the file included with reborn will be returned.

    *Note: After you download a file for the first time, the downloaded file path will be returned on the next call to
    this function to avoid downloading the same file multiple times.*

    Arguments:
        pdb_id (str): For example: "101M" or "101M.pdb".  There are also some special strings which so far include
                         "lysozyme" (2LYZ) and "PSI" (1jb0)
        save_path (string): Path to the downloaded file.  The default is the current working directory, which depends
                            on where you run your code.
        silent (bool): Set to true if you do not want any text printed to stdout.

    Returns:
        string : Path to PDB file
    """

    if pdb_id == 'lysozyme':
        pdb_id = '2LYZ'

    if pdb_id == 'PSI':
        pdb_id = '1jb0'

    if not pdb_id.endswith('.pdb'):
        pdb_id += '.pdb'

    pdb_path = os.path.join(save_path, pdb_id)

    # Check if the file already exists
    if os.path.isfile(pdb_path):
        return pdb_path

    # Check if the file is cached in reborn
    if os.path.exists(pdb_data_path+'/'+pdb_id):
        return pdb_data_path+'/'+pdb_id

    # Finally, try to download from web if all else fails
    if not os.path.isfile(pdb_path):
        pdb_web_path = 'https://files.rcsb.org/download/' + pdb_id
        print('Downloading %s to %s' % (pdb_web_path, pdb_path))
        urllib.request.urlretrieve(pdb_web_path, pdb_path)
        return pdb_path

    return None


class UnitCell(object):
    r"""
    Simple class for *primitive* unit cell information.  Provides the convenience methods r2x(), x2r(), q2h() and h2q()
    for transforming between fractional and orthogonal coordinates in real space and reciprocal space.

    .. note::

        We do not handle non-primitive unit cells in reborn.  We always assume that the unit cell contains one and only
        one crystal lattice point per cell.
    """

    a = None  #: Lattice constant (float)
    b = None  #: Lattice constant (float)
    c = None  #: Lattice constant (float)
    alpha = None  #: Lattice angle in radians (float)
    beta = None  #: Lattice angle in radians (float)
    gamma = None  #: Lattice angle in radians (float)
    volume = None  #: Unit cell volume (float)
    o_mat = None  #: Orthogonalization matrix (3x3 array).  Does the transform r = O.x on fractional coordinates x.
    o_mat_inv = None  #: Inverse orthogonalization matrix (3x3 array)
    a_mat = None  #: Orthogonalization matrix transpose (3x3 array). Does the transform q = A.h, with Miller indices h.
    a_mat_inv = None  #: A inverse

    def __init__(self, a, b, c, alpha, beta, gamma):
        r"""
        Always initialize with the lattice parameters.  Units are SI and radians.

        FIXME: Allow initialization by other specifications

        Arguments:
            a: Lattice constant
            b: Lattice constant
            c: Lattice constant
            alpha: Lattice angle
            beta:  Lattice angle
            gamma: Lattice angle
        """

        al = alpha
        be = beta
        ga = gamma

        self.a = a
        self.b = b
        self.c = c
        self.alpha = al
        self.beta = be
        self.gamma = ga

        vol = a * b * c * np.sqrt(1 - np.cos(al)**2 - np.cos(be) **
              2 - np.cos(ga)**2 + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
        o_mat = np.array([
                [a, b * np.cos(ga), c * np.cos(be)],
                [0, b * np.sin(ga), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
                [0, 0, vol / (a * b * np.sin(ga))]
                ])
        o_inv = np.array([
                [1 / a, -np.cos(ga) / (a * np.sin(ga)), 0],
                [0, 1 / (b * np.sin(ga)), 0],
                [0, 0, a * b * np.sin(ga) / vol]
                ])
        self.o_mat = o_mat
        self.o_mat_inv = o_inv
        self.a_mat = o_inv.T.copy()
        self.a_mat_inv = o_mat.T.copy()
        self.volume = vol

    def __str__(self):
        s  = 'UnitCell\n'
        s += '========\n'
        s += '(a, b, c) = (%.3g, %.3g, %.3g)\n' % (self.a, self.b, self.c)
        s += '(alpha, beta, gamma) = (%.3g, %.3g, %.3g)\n' % (self.alpha, self.beta, self.gamma)
        s += 'Orth. Matrix:\n'
        s += self.o_mat.__str__()
        s += '\n--------\n'
        return s

    def r2x(self, r_vecs):
        r""" Transform orthogonal coordinates to fractional coordinates. """
        return np.dot(r_vecs, self.o_mat_inv.T)

    def x2r(self, x_vecs):
        r""" Transform fractional coordinates to orthogonal coordinates. """
        return np.dot(x_vecs, self.o_mat.T)

    def q2h(self, q_vecs):
        r""" Transform reciprocal coordinates to fractional Miller coordinates. """
        return np.dot(q_vecs, self.a_mat_inv.T)/2./np.pi

    def h2q(self, h_vecs):
        r""" Transform fractional Miller coordinates to reciprocal coordinates. """
        return 2.*np.pi*np.dot(h_vecs, self.a_mat.T)

    @property
    def a_vec(self):
        r""" Crystal basis vector a. """
        return self.o_mat[:, 0].copy()

    @property
    def b_vec(self):
        r""" Crystal basis vector b. """
        return self.o_mat[:, 1].copy()

    @property
    def c_vec(self):
        r""" Crystal basis vector c. """
        return self.o_mat[:, 2].copy()

    @property
    def as_vec(self):
        r""" Reciprocal basis vector a*. """
        return self.a_mat[:, 0].copy()

    @property
    def bs_vec(self):
        r""" Reciprocal basis vector b*. """
        return self.a_mat[:, 1].copy()

    @property
    def cs_vec(self):
        r""" Reciprocal basis vector c*. """
        return self.a_mat[:, 2].copy()


class SpaceGroup(object):
    r"""
    Container for crystallographic spacegroup information.  Most importantly, transformation matices and vectors.  These
    transformations are purely in the fractional coordinate basis.  Note that this class has no awareness of the
    meanings of spacegroup symbols -- I have not yet found a good way to programmatically go from a spacegroup symbol
    string to a set of symmetry operators.
    """

    spacegroup_symbol = None  #: Spacegroup symbol (free format... has no effect)
    sym_rotations = None      #: List of 3x3 transformation matrices (fractional coordinates)
    sym_translations = None   #: List of symmetry translation vectors (fractional coordinates)

    def __init__(self, spacegroup_symbol, sym_rotations, sym_translations):
        r"""
        Initialization requires that you determine the lists of rotations and translations yourself and provide them
        upon instantiation.
        """
        self.spacegroup_symbol = spacegroup_symbol
        self.sym_rotations = sym_rotations
        self.sym_translations = sym_translations

    def __str__(self):

        s = 'SpaceGroup\n'
        s += '==========\n'
        if self.spacegroup_symbol is not None:
            s += self.spacegroup_symbol + '\n'
        for k in range(len(self.sym_rotations)):
            s += 'operation %d\n' % (k,)
            s += self.sym_translations[k].__str__() + '\n'
            s += self.sym_rotations[k].__str__() + '\n'
        s += '==========\n'
        return s

    @property
    def n_molecules(self):
        r""" The number of symmetry operations. """
        return self.n_operations

    @property
    def n_operations(self):
        r""" The number of symmetry operations. """
        return len(self.sym_rotations)

    def apply_symmetry_operation(self, op_num, x_vecs):
        r"""
        Apply a symmetry operation to an asymmetric unit.

        Arguments:
            op_num (int): The number of the operation to be applied.
            x_vecs (Nx3 array): The atomic coordinates in the crystal basis.

        Returns: Nx3 array.
        """
        rot = self.sym_rotations[op_num]
        trans = self.sym_translations[op_num]
        return np.dot(x_vecs, rot.T) + trans

    def apply_inverse_symmetry_operation(self, op_num, x_vecs):
        r"""
        Apply an inverse symmetry operation to an asymmetric unit.

        Arguments:
            op_num (int): The number of the operation to be applied.
            x_vecs (Nx3 array): The atomic coordinates in the crystal basis.

        Returns: Nx3 array.
        """
        rot = self.sym_rotations[op_num]
        trans = self.sym_translations[op_num]
        return np.dot((x_vecs - trans), np.linalg.inv(rot).T)


class CrystalStructure(object):
    r"""
    A container class for stuff needed when dealing with crystal structures.  Combines information regarding the
    asymmetric unit, unit cell, and spacegroup.
    """
    # TODO: Needs documentation!

    fractional_coordinates = None  #: Fractional coords of the asymmetric unit, possibly with NCS partners
    molecule = None                #: Molecule class instance containing the asymmetric unit
    unitcell = None                #: UnitCell class instance
    spacegroup = None              #: Spacegroup class instance
    mosaicity_fwhm = 0
    crystal_size = 1e-6
    crystal_size_fwhm = 0.0
    mosaic_domain_size = 0.5e-6
    mosaic_domain_size_fwhm = 0.0
    cryst1 = ""
    pdb_dict = None
    _au_com = None

    def __init__(self, pdb_file_path, no_warnings=False, expand_ncs_coordinates=False, tight_packing=False,
                       unitcell=None, spacegroup=None):
        r"""
        This class is initialized with a PDB file.

        FIXME: Allow initialization without a PDB file.

        Arguments:
            pdb_file_path (string): Path to a pdb file (or just PDB ID if you want it auto-downloaded)
            no_warnings (bool): Suppress warnings concerning ambiguities in symmetry operations
            expand_ncs_coordinates (bool): Choose whether or not to expand non-crystallographic symmetry (NCS) partners.
            tight_packing (bool): Choose whether or not to enable a physical arrangement of the asymmetric unit symmetry partners - No difference for inifinite crystals but makes sense for a finite crystal.
            unitcell (Unit Cell class): Specify a unit cell manually if needed.
            spacegroup (Space Group class): Specify a space group manually if needed.
        """

        if not os.path.exists(pdb_file_path):
            # print("PDB file not found.  Attempting to download it to %s." % temp_dir)
            pdb_file_path = get_pdb_file(pdb_file_path, save_path=temp_dir)

        dic = pdb_to_dict(pdb_file_path)
        self.pdb_dict = dic

        a, b, c, al, be, ga = dic['unit_cell']

        if unitcell is not None:
            # Use the user-specified unit cell
            self.unitcell = unitcell
            S = self.unitcell.o_mat_inv * 1e-10 # Convert to meters
            U = np.zeros(3)
        else:
            if (a == 1) and (b == 1) and (c == 1):
                # This is the case for the PDBs that are deposited from a single particle electron microscopy experiment,
                # where the unit cell constants are set artificially to a = b = c = 1 Ang 
                # and the angles al = be = ga = 90 degrees.

                # We will set the unit cell to be the difference between the max and min of the atomic coordinates in the three directions.
                r_min = np.zeros(3)
                r_min[0] = np.min(dic['atomic_coordinates'][:, 0])
                r_min[1] = np.min(dic['atomic_coordinates'][:, 1])
                r_min[2] = np.min(dic['atomic_coordinates'][:, 2])

                r_max = np.zeros(3)
                r_max[0] = np.max(dic['atomic_coordinates'][:, 0])
                r_max[1] = np.max(dic['atomic_coordinates'][:, 1])
                r_max[2] = np.max(dic['atomic_coordinates'][:, 2])

                self.unitcell = UnitCell((r_max[0] - r_min[0])*1e-10, (r_max[1] - r_min[1])*1e-10, (r_max[2] - r_min[2])*1e-10, 90*np.pi/180, 90*np.pi/180, 90*np.pi/180)
                S = self.unitcell.o_mat_inv * 1e-10 # Convert to meters
                U = np.zeros(3)
            else:
                # Set the unit cell to be the one from the PDB (vast majority of PDBs will execute this code).
                self.unitcell = UnitCell(a*1e-10, b*1e-10, c*1e-10, al*np.pi/180, be*np.pi/180, ga*np.pi/180)
                S = dic['scale_matrix']
                U = dic['scale_translation']

        if np.sum(np.abs(U)) > 0:
            if not no_warnings:
                warn('\nThe U vector is not equal to zero, which could be a serious problem.  Look here:\n'
                     'https://rkirian.gitlab.io/reborn/crystals.html.\n')

        # These are the initial coordinates with strange origin
        r = dic['atomic_coordinates']
        
        Z0 = atoms.atomic_symbols_to_numbers(dic['atomic_symbols'])
        atomic_symbols0 = dic['atomic_symbols']

        # Initialisations for generating NCS partners
        Z = Z0.copy()
        atomic_symbols = atomic_symbols0.copy()

        # Check for non-crystallographic symmetry.  Construct asymmetric unit from them.
        if expand_ncs_coordinates:
            ncs_partners = [r]
            n_ncs_partners = len(dic['ncs_rotations'])
            i_given = dic['i_given']
            for i in range(n_ncs_partners):
                if i_given[i] == 0:
                    Z = np.concatenate([Z0, Z])
                    atomic_symbols = np.concatenate([atomic_symbols0, atomic_symbols])
                    R = dic['ncs_rotations'][i]
                    T = dic['ncs_translations'][i]
                    ncs_partners.append(np.dot(r, R.T) + T)
            r_au = np.vstack(ncs_partners)
        else:
            r_au = r

        # Transform to fractional coordinates
        x_au = np.dot(S, r_au.T).T # + U

        # Get center of mass (estimate based on atomic number only)
        x_au_com = np.sum((Z*x_au.T).T, axis=0)/np.sum(Z)

        n_sym = len(dic['spacegroup_rotations'])
        rotations = []
        translations = []
        for i in range(n_sym):
            R = dic['spacegroup_rotations'][i]
            T = dic['spacegroup_translations'][i]
            W = np.dot(S, np.dot(R, np.linalg.inv(S)))
            # assert np.max(np.abs(W - np.round(W))) < 5e-2  # 5% error OK?
            W = np.round(W)
            Z = np.dot(S, T) # + np.dot(np.eye(3)-W, U)
            # assert np.max(np.abs(Z - np.round(Z*12)/12)) < 5e-2  # 5% error OK?
            Z = np.round(Z*12)/12
            rotations.append(W)
            translations.append(Z)

        if spacegroup is not None:
            self.spacegroup = spacegroup
        else:
            self.spacegroup = SpaceGroup(dic['spacegroup_symbol'], rotations, translations)

        self.fractional_coordinates = x_au
        self.fractional_coordinates_com = x_au_com

        # Redefine spacegroup operators so that all molecule COMs are in the unit cell
        if tight_packing:
            self.set_tight_packing()

        r_au_mod = np.dot(x_au, self.unitcell.o_mat.T)
        self.molecule = Molecule(coordinates=r_au_mod, atomic_symbols=atomic_symbols)

    @property
    def x_vecs(self):
        r"""
        Fractional coordinates of atoms.

        Returns: Nx3 numpy array
        """

        return self.fractional_coordinates

    @property
    def x(self):
        return self.x_vecs

    def get_symmetry_expanded_coordinates(self):
        r"""

        Returns: All atomic coordinates including spacegroup symmetry partners.

        """
        x0 = self.x
        xs = [] # Initialise an empty array to hold the symmetry partner coordinates
        for (R, T) in zip(self.spacegroup.sym_rotations, self.spacegroup.sym_translations):
            xs.append(np.dot(x0, R.T) + T)
        # Concatenate and return the coordinates in realspace (orthogonal coordinates)
        return np.dot(np.concatenate(xs), self.unitcell.o_mat.T)

    def set_tight_packing(self):
        r"""
        Re-define the atomic coordinates and symmetry transforms so that the center-of-mass of each molecule lies within
        the unit cell.  Two steps: first shift the coordinates so that the COM of the asymmetric unit lies in the unit
        cell.  Next, add integral shifts to each of the symmetry operations as needed to put the COM of all symmetry
        partners into the unit cell.

        Returns: None
        """
        # First shift the asymmetric unit so that it's COM lies in the unit cell.  Sometimes it does not.
        com = self.fractional_coordinates_com
        self.fractional_coordinates -= com - (com % 1)
        self.fractional_coordinates_com -= com - (com % 1)
        # Now re-define the symmetry operations so that all symmetry partners also lie in the unit cell.
        for i in range(self.spacegroup.n_molecules):
            com = self.spacegroup.apply_symmetry_operation(i, self.fractional_coordinates_com)
            self.spacegroup.sym_translations[i] -= com - (com % 1)


class FiniteLattice(object):
    r"""
    A utility for creating finite crystal lattices.  Uses an occupancy model in which a maximum-size array of
    occupancies is created and set to 1 (occupied), and subsequently facets may be added by specifying the plane
    at which the cut is made.  A cut sets the occupancies beyond the plane to 0 (unoccupied).  Lattice vector positions
    can then be generated in the crystal or cartesian basis.  Gaussian disorder may be added.  Special shapes are
    supported, including hexagonal prisms, parallelepipeds, and spheres.
    """

    def __init__(self, max_size=None, unitcell=None):
        r"""
        Arguments:

            max_size: Integer N that sets the size of the lattice to N x N x N.
            unitcell: A crystal.UnitCell type that is needed to generate
        """

        if max_size is None:
            raise ValueError("You need to choose a maximum lattice size.")

        if not isinstance(unitcell, UnitCell):
            raise ValueError("You must provide a unitcell of crystal.UnitCell type.")

        if not isinstance(max_size, int):
            raise ValueError("max_size must be an int.")

        if max_size <= 0:
            raise ValueError("max_size must be >= 0.")

        self.max_size = max_size
        self.occupancies = np.ones([self.max_size]*3)
        self.unitcell = unitcell
        ran = np.arange(max_size) - np.floor((max_size-1)/2)  # e.g. [-1, 0, 1, 2] or [-1, 0, 1]
        x, y, z = np.meshgrid(ran, ran, ran, indexing='ij')
        self.all_x_coordinates = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T.copy()
        self.all_x_coordinates.flags.writeable = False
        self._all_r_coordinates = None
        self._all_r_mags = None
        self.sigmas = None
        self.disordered = False

    @property
    def __all_r_coordinates(self):

        if self._all_r_coordinates is None:
            self._all_r_coordinates = np.dot(self.all_x_coordinates, self.unitcell.o_mat.T)
        return self._all_r_coordinates

    @property
    def __all_r_mags(self):

        if self._all_r_mags is None:
            self._all_r_mags = vec_mag(self.__all_r_coordinates)
        return self._all_r_mags

    @property
    def __occupied_indices(self):

        return np.where(self.occupancies.ravel() != 0)[0]

    @property
    def occupied_x_coordinates(self):
        r"""
        The occupied coordinate vectors in the crystal basis.  An (M, 3) numpy array.
        """
        x = self.all_x_coordinates[self.__occupied_indices, :]
        if self.disordered:
            x += np.random.normal(size=x.shape)*self.sigmas
        return x

    @property
    def occupied_r_coordinates(self):
        r"""
        The occupied coordinate vectors in the cartesian (laboratory) basis.  An (M, 3) numpy array.
        """
        return np.dot(self.occupied_x_coordinates, self.unitcell.o_mat.T)

    def add_facet(self, plane=None, length=None, shift=0):
        r"""
        Creates a crystal facet by zeroing the lattice occupancies for which the following condition is met:

        .. math::

            (\mathbf{x}+\mathbf{s})\cdot\mathbf{p} > L

        where :math:`\mathbf{x}` are the nominal lattice positions (with one lattice point sitting on the origin),
        :math:`\mathbf{s}` is an optional shift (which might correspond to the center of mass of one molecule
        symmetry partner in a spacegroup), :math:`\mathbf{p}` defines the facet orientation (a vector that points from
        the origin to the normal of the facet surface), and :math:`L` is the length from the origin to the facet
        surface.

        This operation will set the occupancies of the rejected lattice sites to zero.  When you subsequently access
        the :attr:`occupied_x_coordinates <reborn.target.crystal.FiniteLattice.occupied_x_coordinates>` property,
        the zero-valued occupancies will not be returned.

        Arguments:

            plane (numpy array):  The vector :math:`\vec{p}` defined above.
            length (numpy array):  The length :math:`L` defined above.
            shift (numpy array):  The vector :math:`\vec{s}` defined above.
        """

        proj = (self.all_x_coordinates + shift).dot(np.array(plane))
        w = np.where(proj > length)[0]
        if len(w) > 0:
            self.occupancies.flat[w] = 0

    def reset_occupancies(self):
        r""" Set all occupancies to 1 """
        self.occupancies = np.ones([self.max_size]*3)

    def make_hexagonal_prism(self, width=3, length=10, shift=0):
        r"""
        Specialized function to form a hexagonal prism by adding eight facets.  A crude illustration is shown
        below.  This method assumes a "standard" hexagonal cell in which alpha=90, beta=90, gamma=120.  The length and
        width parameters span from facet to facet.  The length facets are in the planes [0,0,1] and [0,0,-1].  The three
        widths specify the facet pairs ([1,0,0], [-1,0,0]), ([0,1,0], [0,-1,0]),  ([1,-1,0], [-1,1,0]).  Note that, by
        default, there is always at minimum one lattice point that lies on the origin; if that is not desired then you
        may use the shift parameter as discussed in the
        :meth:`add_facet <reborn.target.crystal.FiniteLattice.add_facet>` method.

        Arguments:
            width (float or array): Three widths to specify the prism shape/size, as explained above.
            length (float): Length of the prism, as illustrated above.
            shift (numpy array):  An optional shift of the entire crystal (defaults to 0).
        """

        width = np.array(width).squeeze()
        if width.size == 1:
            width = np.ones(3)*width
        elif width.size != 3:
            raise ValueError('width must be either float, int or 3-element array')

        self.reset_occupancies()
        self.add_facet(plane=[1, 0, 0], length=width[0]/2, shift=shift)
        self.add_facet(plane=[-1, 0, 0], length=width[0]/2, shift=shift)
        self.add_facet(plane=[0, 1, 0], length=width[1]/2, shift=shift)
        self.add_facet(plane=[0, -1, 0], length=width[1]/2, shift=shift)
        self.add_facet(plane=[-1, 1, 0], length=width[2]/2, shift=shift)
        self.add_facet(plane=[1, -1, 0], length=width[2]/2, shift=shift)
        self.add_facet(plane=[0, 0, 1], length=length/2, shift=shift)
        self.add_facet(plane=[0, 0, -1], length=length/2, shift=shift)

    def make_parallelepiped(self, shape=(5, 5, 5), shift=0):
        r""" Cuts out a Parallelepiped shaped crystal.

        Arguments:
            shape (3-tuple):  Number of lattice points to include in the parallelepiped in each direction.
            shift (numpy array):  An optional shift of the entire crystal (defaults to 0).
        """

        self.reset_occupancies()
        self.add_facet(plane=[1, 0, 0],  length=shape[0]/2, shift=shift)
        self.add_facet(plane=[-1, 0, 0], length=shape[0]/2, shift=shift)
        self.add_facet(plane=[0, 1, 0],  length=shape[1]/2, shift=shift)
        self.add_facet(plane=[0, -1, 0], length=shape[1]/2, shift=shift)
        self.add_facet(plane=[0, 0, 1],  length=shape[2]/2, shift=shift)
        self.add_facet(plane=[0, 0, -1], length=shape[2]/2, shift=shift)

    def make_sphere(self, radius):
        r"""
        Create a spherical crystal, using the radius in cartesian space.

        Arguments:
            radius (float):  The cartesian-space radius.
        """
        self.occupancies.flat[self.__all_r_mags > radius] = 0

    def set_gaussian_disorder(self, sigmas=(0.0, 0.0, 0.0)):
        r"""
        Add Gaussian-distributed random offsets to all lattice points

        Arguments:
            sigmas (numpy array):  The three standard deviations along crystal basis vectors (so these should probably
                                   be less than 1)
        """
        sigmas = np.array(sigmas).squeeze()
        if np.sum(np.abs(sigmas)) == 0:
            self.disordered = False
        else:
            self.disordered = True
        self.sigmas = sigmas


class CrystalDensityMap(object):
    r"""
    A helper class for working with 3D crystal density maps.  Most importantly, it helps with spacegroup symmetry
    transformations.  Once initialized with a crystal spacegroup and lattice, along with desired resolution
    (:math:`1/d`) and oversampling ratio (equal to 1 for normal crystallographic Bragg sampling), this tool chooses the
    shape of the map and creates lookup tables for the symmetry transformations.  The shape is chosen according to
    the formula :math:`(N_1, N_2, N_3) = \text{roundup}(a_1/d, a_2/d, a_3/d)` where the function
    :math:`\texttt{roundup}` takes the next-largest integer multiple of :math:`P`.  The value of :math:`P` comes from
    the point group and is equal to the largest number of rotations about any axis.  For example, the spacegroup
    :math:`P6_3` has a point group of 6, and thus :math:`P=6`.

    It is generally assumed that you are working in the crystal basis (fractional coordinates).  When working in the
    crystal basis, the symmetry transforms avoid interpolation artifacts.  The shape of the map is chosen specifically
    to avoid interpolation artifacts.

    This class does not maintain the data array as the name might suggest.  It provides methods needed to work on the
    data arrays.
    """

    sym_luts = None      #: Symmetry lookup tables -- indices that map AU to sym. partner
    cryst = None         #: CrystalStructure class used to initiate the map
    oversampling = None  #: Oversampling ratio
    dx = None            #: Length increments for fractional coordinates
    cshape = None        #: Number of samples along edges of unit cell within density map
    shape = None         #: Number of samples along edge of full density map (includes oversampling)
    size = None          #: Total number of elements in density map (np.prod(self.shape)
    strides = None       #: The stride vector (mostly for internal use)

    def __init__(self, cryst, resolution, oversampling):
        r"""
        On initialization, you provide a CrystalStructure class instance, along with your desired resolution
        and oversampling.

        Arguments:
            cryst (:class:`CrystalStructure` instance) : A crystal structure that contains the spacegroup and lattice
                                                      information.
            resolution (float) : The desired resolution of the map.  The resolution :math:`d` is defined such that the
              number of samples along the :math:`n`__th edge is greater than or equal to :math:`a_n/d` for lattice
              constant :math:`a_n`.  The actual value will be rounded up toward the nearest multiple of 1, 2, 3, 4, or 6
              (whichever is appropriate for the spacegroup symmetry).
            oversampling (int) : An oversampling of 2 gives a real-space map that is twice as large as the unit cell. In
                                  Fourier space, there will be one sample between Bragg samples.  And so on for 3,4,...
        """

        # Given desired resolution and unit cell, these are the number of voxels along each edge of unit cell.
        # cshape = np.ceil((1/resolution) * (1/vec_mag(cryst.unitcell.a_mat.T)))
        cshape = np.ceil(vec_mag(cryst.unitcell.o_mat.T)/resolution)

        # The number of samples along an edge must be a multiple of the shortest translation.  E.g., if an operation
        # consists of a translation of 1/3 or 2/3 distance along the cell, the shape must be a multiple of 3.
        v = np.ravel(cryst.spacegroup.sym_translations)
        v[v == 0] = 1
        for m in [1, 2, 3, 4, 6]:
            if np.sum(np.abs(np.round(v*m) - v*m)) == 0:
                break
        cshape = np.ceil(cshape / m) * m

        self.cryst = cryst
        self.oversampling = np.int64(np.ceil(oversampling))
        self.dx = 1.0 / cshape
        self.cshape = cshape.astype(int)
        self.shape = (cshape * self.oversampling).astype(int)
        self.size = np.prod(self.shape)
        self.strides = np.array([self.shape[2]*self.shape[1], self.shape[2], 1])

    def __str__(self):
        s = 'CrystalDensityMap\n'
        s += '\tOversampling = %d\n' % self.oversampling
        s += '\tUnit cell shape = (%d, %d, %d)\n' % tuple(self.cshape)
        s += '\tPadded shape = (%d, %d, %d)\n' % tuple(self.shape)
        s += '\tdx = (%.3g, %.3g, %.3g)\n' % tuple(self.dx)
        return s

    @property
    def n_vecs(self):
        r"""
        Get an Nx3 array of vectors corresponding to the indices of the map voxels.  The array looks like this:

        [[0, 0, 0], [0, 0, 1], [0, 0, 2],  ...  , [Nx-1,Ny-1,Nz-1]]

        Note that it is the third index, which we might associate with "z", that increments most rapidly.

        Returns: numpy array
        """

        shp = self.shape
        ind = np.arange(0, self.size)
        n_vecs = np.zeros([self.size, 3])
        n_vecs[:, 0] = np.floor(ind / (shp[1]*shp[2]))
        n_vecs[:, 1] = np.floor(ind / shp[2]) % shp[1]
        n_vecs[:, 2] = ind % shp[2]
        return n_vecs

    @property
    def x_vecs(self):
        r"""
        Get an Nx3 array that contains the fractional coordinates.  For example, if there were four samples per unit
        cell, the array will look like this:

        [[0, 0, 0], [0, 0, 0.25], [0, 0, 0.5], [0, 0, 0.75], [0, 0.25, 0], ... , [ 0.75, 0.75, 0.75]]

        Returns: numpy array
        """

        return self.n_vecs * self.dx

    @property
    def x_limits(self):
        r"""
        Return a 3x2 array with the limits of the density map.  These limits correspond to the concentions described
        in the :ref:`documentation <working_with_maps>` ; :math:`x_\text{min} = \text{x_limits[:, 0]}` .

        Returns:
        """

        shp = self.shape
        dx = self.dx
        return np.array([[0, dx[0]*shp[0]], [0, dx[1]*shp[1]], [0, dx[2]*shp[2]]])

    @property
    def x_min(self):
        return np.zeros(3)

    @property
    def x_max(self):
        return (self.shape - 1) * self.dx

    @property
    def h_vecs(self):
        r"""
        This provides an :math:`M \times 3` array of Fourier-space vectors :math:`\vec{h}` that correspond to the
        real-space vectors :math:`\vec{x}` of this density map.  These :math:`\vec{h}` vectors can be understood as the
        "fractional Miller indices", and they are defined in accordance with the
        `numpy FFT convention <https://docs.scipy.org/doc/numpy/reference/routines.fft.html>`_ .
        The numpy foward FFT is defined in these terms as

        .. math::

            F_h = \sum_{x=0}^{N-1} f_x \exp(-2 \pi i h x / N) \; , \quad h = 0, 1, 2, \ldots , N-1

        while the inverse FFT is

        .. math::

            f_x = \frac{1}{N}\sum_{h=0}^{N-1} F_h \exp(2 \pi i h x / N) \; , \quad x = 0, 1, 2, \ldots , N-1

        Note that the above expression is defined such that both :math:`h` and :math:`x` have integer step size.  Our
        :class:`CrystalDensityMap` class handles oversampling, in which case the integer :math:`h` become
        :math:`h \rightarrow h/s` and the array size grows according to :math:`N \rightarrow s N`, where
        :math:`s` is the oversampling ratio.

        Returns: numpy array
        """

        h0 = np.fft.fftshift(np.fft.fftfreq(self.shape[0], d=self.oversampling/self.shape[0]))
        h1 = np.fft.fftshift(np.fft.fftfreq(self.shape[1], d=self.oversampling/self.shape[1]))
        h2 = np.fft.fftshift(np.fft.fftfreq(self.shape[2], d=self.oversampling/self.shape[2]))
        hh0, hh1, hh2 = np.meshgrid(h0, h1, h2, indexing='ij')
        h_vecs = np.empty((self.size, 3))
        h_vecs[:, 0] = hh0.ravel()
        h_vecs[:, 1] = hh1.ravel()
        h_vecs[:, 2] = hh2.ravel()
        return h_vecs

    @property
    def h_limits(self):
        r"""
        This is depreciated.  Do not use it.

        Returns:

        """
        limits = np.zeros((3, 2))
        limits[:, 0] = -np.floor(self.shape/2)/self.oversampling
        limits[:, 1] = np.floor((self.shape-1) / 2)/self.oversampling
        return limits

    @property
    def h_min(self):
        r"""
        The lower limits of the :math:`\vec{h}` vectors, which correspond to the FFT of this density map.

        Returns:
            numpy array
        """
        return -np.floor(self.shape / 2) / self.oversampling

    @property
    def h_max(self):
        r"""
        The upper limits of the :math:`\vec{h}` vectors, which correspond to the FFT of this density map.

        Returns:
            numpy array
        """
        return np.floor((self.shape - 1) / 2) / self.oversampling

    @property
    def q_vecs(self):
        r""" For convenience, converts self.h_vecs to q vectors (using self.cryst.unitcell.h2q method). """
        return self.cryst.unitcell.h2q(self.h_vecs)

    @property
    def q_min(self):
        r""" For convenience, converts self.h_min to a q vector (using self.cryst.unitcell.h2q method).  Note that
        sampling in q space will not match with the sampling in h space unless the unit cell is orthorhombic. """
        return self.cryst.unitcell.h2q(self.h_min)

    @property
    def q_max(self):
        r""" For convenience, converts self.h_max to a q vector (using self.cryst.unitcell.h2q method). """
        return self.cryst.unitcell.h2q(self.h_max)

    def get_sym_luts(self):
        r"""
        This provides a list of "symmetry transform lookup tables" (symmetry LUTs).  These are the flattened array
        indices that map voxels from the asymmetric unit (AU) map to the kth symmetry partner.  This kind of
        transformation, from AU to symmetry partner k, is performed as follows:

        .. code-block:: python

            # data is a 3D array of densities
            data_trans = np.empty_like(data)
            lut = crystal_density_map.get_sym_luts()[k]
            data_trans.flat[lut] = data.flat[:]

        For convenience the above operation may be performed by the method
        :meth:`au_to_k <reborn.target.crystal.CrystalDensityMap.au_to_k>`, while the inverse operation may be
        performed by the method :meth:`k_to_au <reborn.target.crystal.CrystalDensityMap.k_to_au>`.  The method
        :meth:`symmetry_transform <reborn.target.crystal.CrystalDensityMap.symmetry_transform>` can be used to
        transform from one symmetry partner to another.

        Note that the LUTs are kept in memory for future use - beware of the memory requirement.

        Returns:
            list of numpy arrays : The symmetry lookup tables (LUTs)
        """

        if self.sym_luts is None:

            sym_luts = []
            x0 = self.x_vecs

            for (R, T) in zip(self.cryst.spacegroup.sym_rotations, self.cryst.spacegroup.sym_translations):
                lut = np.dot(R, x0.T).T + T          # transform x vectors in 3D grid
                lut = np.round(lut / self.dx)        # switch from x to n vectors
                lut = lut % self.shape               # wrap around
                lut = np.dot(self.strides, lut.T)    # in p space
                assert np.sum(lut - np.round(lut)) == 0
                sym_luts.append(lut.astype(np.int64))
            self.sym_luts = sym_luts

        return self.sym_luts

    def au_to_k(self, k, data):
        r"""
        Transform a map of the asymmetric unit (AU) to the kth symmetry partner.  Note that the generation of the
        first symmetry partner (k=0, where k = 0, 1, ..., N-1) might employ a non-identity rotation matrix and/or a
        non-zero translation vector -- typically this is not the case but it can happen for example if the symmetry
        operations are chosen such that all molecules are packed within the unit cell.

        Arguments:
            k (int) : The index of the symmetry partner (starting with k=0)
            data (|ndarray|) : The input data array.

        Returns:
            |ndarray| : Transformed array
        """
        # data_out = np.empty_like(data)
        # lut = self.get_sym_luts()[k]
        # data_out.flat[lut] = data.flat[:]
        data_out = self.symmetry_transform(0, k, data)
        return data_out

    def k_to_au(self, k, data):
        r"""
        Transform a map of the kth symmetry partner to the asymmetric unit.  This reverses the action of the
        :meth:`au_to_k <reborn.target.crystal.CrystalDensityMap.au_to_k>` method.

        Args:
            k (int) : The index of the symmetry partner (starting with k=0)
            data (3D |ndarray|) : The input data array.

        Returns:
            3D |ndarray| : Transformed array
        """
        # lut = self.get_sym_luts()[k]
        # data_out = data.flat[lut].reshape(data.shape)
        data_out = self.symmetry_transform(k, 0, data)
        return data_out

    def symmetry_transform(self, i, j, data):
        r"""
        Apply crystallographic symmetry transformation to a density map (3D |ndarray|).  This applies the mapping from
        symmetry element i to symmetry element j, where i=0,1,...,N-1 for a spacegroup with N symmetry operations.

        Arguments:
            i (int) : The "from" index; symmetry transforms are performed from this index to the j index
            j (int) : The "to" index; symmetry transforms are performed from the i index to this index
            data (|ndarray|) : The 3D block of data to apply the symmetry transform on

        Returns: |ndarray| with transformed densities
        """

        luts = self.get_sym_luts()
        data_trans = np.zeros_like(data)
        data_trans.flat[luts[j]] = data.flat[luts[i]]

        return data_trans

    def place_atoms_in_map(self, atom_x_vecs, atom_fs, mode='trilinear', fixed_atom_sigma=0.5e-10):
        r"""
        This will take a list of atom position vectors and densities and place them in a 3D map.  The position vectors
        should be in the crystal basis, and the densities must be real.

        FIXME: This function will soon be replaced with one that utilizes atomic form factors.

        Arguments:
            atom_x_vecs      (numpy array) : An nx3 array of position vectors
            atom_fs          (numpy array) : An n-length array of densities (must be real)
            mode             (str)         : Either 'nearest', 'trilinear', or 'gaussian' (default: 'trilinear')
            fixed_atom_sigma (float)       : Standard deviation of the Gaussian atoms

        Returns:
            numpy array : The sum of densities that were provided as input.
        """

        if mode == 'gaussian':
            raise ValueError('The "gaussian" mode does not implement periodic boundaries properly...')  # FIXME
            sigma = fixed_atom_sigma  # Gaussian sigma (i.e. atom "size"); this is a fudge factor and needs to be
            # updated n_atoms = atom_x_vecs.shape[0]
            orth_mat = self.crystal_structure.unitcell.o_mat.copy()
            map_x_vecs = self.x_vecs
            n_map_voxels = map_x_vecs.shape[0]
            f_map = np.zeros([n_map_voxels], dtype=np.complex)
            f_map_tmp = np.zeros([n_map_voxels], dtype=np.double)
            if len(atom_x_vecs.shape) == 1:
                atom_x_vecs = np.expand_dims(atom_x_vecs, axis=0)
            place_atoms_in_map(atom_x_vecs, atom_fs, sigma, self.x_max, orth_mat, map_x_vecs, f_map, f_map_tmp)
            return np.reshape(f_map, self.shape)

        elif mode == 'trilinear':
            # Note that we do not divide by weightout because we want the sum of the atoms not the mean.
            rho, _ = trilinear_insert(data_coord=np.ascontiguousarray(atom_x_vecs), data_val=atom_fs,
                                      x_min=self.x_min, x_max=self.x_max, n_bin=self.shape, boundary_mode="periodic",
                                      mask=np.full(len(atom_fs), True, dtype=bool))
            return rho

        elif mode == 'nearest':
            rho = np.zeros(self.shape, dtype=atom_fs.dtype)
            dx = (self.x_max - self.x_min)/(self.shape - 1)
            indices = (np.round((atom_x_vecs - self.x_min)/dx) % self.shape).astype(int)
            for i in range(len(atom_fs)):
                xi = indices[i, 0]
                yi = indices[i, 1]
                zi = indices[i, 2]
                rho[xi, yi, zi] += atom_fs[i]
            return rho

    @property
    def voxel_volume(self):
        r"""
        Get the volume of a voxel in orthogonal coordinates (i.e. Cartesian).

        Returns:
            float: Voxel volume
        """
        omat = self.cryst.unitcell.o_mat.copy()
        dx = self.dx.copy()
        v = omat*dx
        vol = np.abs(np.dot(v[:, 0], np.cross(v[:, 1], v[:, 2])))
        return vol


@jit(nopython=True)
def build_atomic_scattering_density_map(x_vecs, f, sigma, x_min, x_max, shape, orth_mat, n_sigma):

    # Note that we must deal with wrap-around when calculating distances from atoms to grid points.
    #
    #
    # bins   |_____*_____|_x___*_____|_____*_____|
    #
    # index        0           1           2           3           4           5           6           7           8
    #
    # wrapped idx  0           1           2           0           1           2           0           1           2
    #
    # The above schematic is for a map with 3 bins.  The grid samples that correspond to x_min and x_max are in the
    # centers of the bins, indicated by the * symbol.  Supposing we want to place a Gaussian centered at the x position,
    # we need to calculate distances to sample points indexed with 0, 1, 2 but with wrap-around factored in.

    n_atoms = f.shape[0]  # Number of atoms
    dx = (x_max - x_min)/(shape - 1)  # Bin width
    b_tot = x_max - x_min + dx  # Total width of bins, including half-bins that extend beyond bin center points
    b_min = x_min - dx/2  # Bin lower bound - extends half a bin beyond bin center point
    b_max = x_max + dx/2  # Bin upper bound - extends half a bin beyond bin center point
    sum_map = np.zeros(shape, dtype=f.dtype)
    sum_map_temp = np.zeros(shape, dtype=f.dtype)

    for n in range(n_atoms):
        x_atom = x_vecs[n, :]
        sum_val = 0
        for i in range(shape[0]):
            xg = x_min[0] + i * dx[0]
            for j in range(shape[1]):
                yg = x_min[1] + j * dx[1]
                for k in range(shape[2]):
                    zg = x_min[2] + j * dx[2]
                    x_grid = np.array([xg, yg, zg])
                    diff = x_grid - x_atom
                    diff = ((diff + b_tot/2) % b_tot) - b_tot/2
                    diff = np.dot(diff, orth_mat.T)
                    val = np.exp(-np.sum(diff**2)/(2*sigma**2))
                    sum_val += val
                    sum_map_temp[i, j, k] = val
        sum_map_temp /= sum_val
        sum_map_temp *= f[n]
        sum_map += sum_map_temp


@jit(nopython=True)
def place_atoms_in_map(x_vecs, atom_fs, sigma, s, orth_mat, map_x_vecs, f_map, f_map_tmp):
    r"""
    Needs documentation...
    """
    # try:
    #     a = s.shape
    # except:
    #     s = np.array([s, s, s])
    n_atoms = x_vecs.shape[0]
    n_map_voxels = map_x_vecs.shape[0]
    # f_map = np.empty([n_map_voxels], dtype=atom_fs.dtype)
    # f_map_tmp = np.empty([n_map_voxels], dtype=x_vecs.dtype)
    for n in range(n_atoms):
        x = x_vecs[n, 0] % s[0]
        y = x_vecs[n, 1] % s[1]
        z = x_vecs[n, 2] % s[2]
        w_tot = 0
        for i in range(n_map_voxels):
            mx = map_x_vecs[i, 0]
            my = map_x_vecs[i, 1]
            mz = map_x_vecs[i, 2]
            dx = np.abs(x - mx)
            dy = np.abs(y - my)
            dz = np.abs(z - mz)
            dx = min(dx, s[0] - dx)
            dy = min(dy, s[1] - dy)
            dz = min(dz, s[2] - dz)
            dr2 = (orth_mat[0, 0] * dx + orth_mat[0, 1] * dy + orth_mat[0, 2] * dz)**2 + \
                  (orth_mat[1, 0] * dx + orth_mat[1, 1] * dy + orth_mat[1, 2] * dz)**2 + \
                  (orth_mat[2, 0] * dx + orth_mat[2, 1] * dy + orth_mat[2, 2] * dz)**2
            w = np.exp(-dr2/(2*sigma**2))
            f_map_tmp[i] = w
            w_tot += w
        f_map += atom_fs[n] * f_map_tmp/w_tot


def pdb_to_dict(pdb_file_path, ignore_waters=False):
    r"""Return a dictionary with a subset of PDB information.  If there are multiple atomic
    models, only the first will be extracted.  Units are the standard PDB units: angstrom and degrees.

    Arguments:
        pdb_file_path: Path to pdb file
        ignore_waters: Ignore water molecules if True.  (Default: False)

    Returns:
        FIXME: Rick: Think about better formatting of returns.
        A dictionary with the following keys:
        - 'scale_matrix' (from SCALE)
        - 'scale_translation' (from SCALE)
        - 'atomic_coordinates'  (from ATOM or HETATM)
        - 'atomic_symbols'  (from ATOM or HETATM)
        - 'unit_cell'  (from CRYST1)
        - 'spacegroup_symbol'  (from CRYST1)
        - 'spacegroup_rotations'  (from SMTRY)
        - 'spacegroup_translations'  (from SMTRY)
        - 'ncs_rotations'  (from MTRIX)
        - 'ncs_translations'  (from MTRIX)
    """

    atomic_coordinates = np.zeros([10000, 3])
    atomic_symbols = []
    scale = np.zeros([3, 4])
    smtry = np.zeros([1000, 6])
    mtrix = np.zeros([10000, 4])
    i_given = np.zeros([1000], dtype=int)
    a = b = c = None
    alpha = beta = gamma = None
    spacegroup_rotations = []
    spacegroup_symbol = None
    spacegroup_translations = []
    poorly_formatted_atoms = False
    smtry_index = 0
    mtrix_index = 0
    atom_index = 0

    with open(pdb_file_path) as pdbfile:

        for line in pdbfile:

            # Check for a model ID; we will only load in the first model for now
            if line[0:5] == 'MODEL':
                model_number = int(line[10:14])
                if model_number > 1:
                    warn('Found more than one atomic model in PDB file.  Keeping only the first one.')
                    break

            # Transformations from orthogonal coordinates to fractional coordinates
            if line[:5] == 'SCALE':
                n = int(line[5]) - 1
                scale[n, 0] = float(line[10:20])
                scale[n, 1] = float(line[20:30])
                scale[n, 2] = float(line[30:40])
                scale[n, 3] = float(line[45:55])

            # The crystal lattice and spacegroup
            if line[:6] == 'CRYST1':
                cryst1 = line
                a = float(cryst1[6:15])
                b = float(cryst1[15:24])
                c = float(cryst1[24:33])
                alpha = float(cryst1[33:40])
                beta = float(cryst1[40:47])
                gamma = float(cryst1[47:54])
                spacegroup_symbol = cryst1[55:66].strip()

            # Spacegroup symmetry operations
            if line[13:18] == 'SMTRY':
                smtry[smtry_index, 0] = float(line[18])
                smtry[smtry_index, 1] = float(line[19:23])
                smtry[smtry_index, 2] = float(line[24:34])
                smtry[smtry_index, 3] = float(line[34:44])
                smtry[smtry_index, 4] = float(line[44:54])
                smtry[smtry_index, 5] = float(line[54:69])
                smtry_index += 1

            # Non-crystallographic symmetry operations
            # TODO: check if we should skip entries with iGiven == 1.  See PDB documentation.  I'm confused.
            if line[0:5] == 'MTRIX':
                mtrix[mtrix_index, 0] = float(line[10:20])
                mtrix[mtrix_index, 1] = float(line[20:30])
                mtrix[mtrix_index, 2] = float(line[30:40])
                mtrix[mtrix_index, 3] = float(line[45:55])
                if len(line) < 60:
                    i_given[mtrix_index] = 0
                else:
                    try:
                        i_given[mtrix_index] = int(line[59])
                    except ValueError:
                        i_given[mtrix_index] = 0

                mtrix_index += 1

            if ignore_waters and ((line[17:20] == "HOH") or (line[17:20] == "TIP")):
                continue

            # Atomic coordinates (these are the orthogonal, cartesian coordinates denoted as r in the docs).
            if line[:6] == 'ATOM  ' or line[:6] == "HETATM":
                atomic_coordinates[atom_index, 0] = float(line[30:38])
                atomic_coordinates[atom_index, 1] = float(line[38:46])
                atomic_coordinates[atom_index, 2] = float(line[46:54])
                at = ''
                if len(line) >= 78:
                    at = line[76:78].strip().capitalize()
                # It is common that the atom type is not in the right place... try something else if so:
                if at == '':
                    at = line[12:14].strip().capitalize()
                    if poorly_formatted_atoms is False:
                        poorly_formatted_atoms = True
                        warn('Poorly formatted PDB.  Atom types might be wrong.')
                atomic_symbols.append(at)

                atom_index += 1
            if atom_index == atomic_coordinates.shape[0]:  # Make the array larger if need be.
                atomic_coordinates = np.vstack([atomic_coordinates, np.zeros((atomic_coordinates.shape[0], 3))])

    spacegroup_rotations = []
    spacegroup_translations = []
    if smtry_index == 0:
        warn('No spacegroup operations found.')
        spacegroup_rotations.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        spacegroup_translations.append(np.array([0, 0, 0]))
    else:
        for i in range(int(smtry_index/3)):
            spacegroup_rotations.append(smtry[i*3:i*3+3, 2:5])
            spacegroup_translations.append(smtry[i * 3:i * 3 + 3, 5])

    ncs_rotations = []
    ncs_translations = []
    i = 0
    for i in range(int(mtrix_index/3)):
        ncs_rotations.append(mtrix[i*3:i*3+3, 0:3])
        ncs_translations.append(mtrix[i*3:i*3+3, 3])
        i_given[i] = i_given[i*3]
    i_given = i_given[:i+1]

    dic = {'scale_matrix': scale[:, 0:3],
           'scale_translation': scale[:, 3],
           'atomic_coordinates': atomic_coordinates[:atom_index, :],
           'atomic_symbols': atomic_symbols[:atom_index],
           'unit_cell': (a, b, c, alpha, beta, gamma),
           'spacegroup_symbol': spacegroup_symbol,
           'spacegroup_rotations': spacegroup_rotations,
           'spacegroup_translations': spacegroup_translations,
           'ncs_rotations': ncs_rotations,
           'ncs_translations': ncs_translations,
           'i_given': i_given
           }

    return dic


class FiniteCrystal(object):

    lattices = None  # : List of :class:`FiniteLattice <reborn.target.crystal.FiniteLattice>` instances.
    cryst = None  # : :class:`CrystalStructure <reborn.target.crystal.CrystalStructure>` instances.
    au_x_coms = None  # : List of numpy arrays that specify center-of-mass coordinates of asymmetric unit and symmetry partners.

    def __init__(self, cryst, max_size=20):
        r"""
        Utility that allows for the shaping of finite crystal lattices, with consideration of the crystal structure
        (molecular structure, spacegroup, etc.).  This is useful for simulating complete crystals with strange edges
        or other defects that depart from idealized crystals.

        Args:
            cryst (:class:`CrystalStructure`) : A crystal structure object.
                           The center-of-mass of asymmetric unit and spacegroup provided by this object will affect the
                           centering of the lattices.
            max_size (3-element array) : Same as in the :class:`FiniteLattice <reborn.target.crystal.FiniteLattice>`
                                         class.
        """
        self.cryst = cryst
        max_size = max_size

        # Center of mass coordinates for the symmetry partner molecules
        self.au_x_coms = [cryst.spacegroup.apply_symmetry_operation(i, cryst.fractional_coordinates_com)
                     for i in range(cryst.spacegroup.n_operations)]

        self.lattices = [FiniteLattice(max_size=max_size, unitcell=cryst.unitcell)
                         for _ in range(cryst.spacegroup.n_molecules)]

    def add_facet(self, plane=None, length=None):
        r"""
        See equivalent method in :class:`FiniteLattice <reborn.target.crystal.FiniteLattice>`. In this case, the
        facet is added to *all* of the finite lattices (one for each symmetry partner).
        """
        for k in range(len(self.lattices)):
            lat = self.lattices[k]
            com = self.au_x_coms[k]
            lat.add_facet(plane=plane, length=length, shift=com)

    def reset_occupancies(self):
        r"""
        See equivalent method in :class:`FiniteLattice <reborn.target.crystal.FiniteLattice>`. In this case, *all*
        occupancies are reset (i.e. for all symemtry partner lattices).
        """
        for k in range(len(self.lattices)):
            self.lattices[k].reset_occupancies()

    def make_hexagonal_prism(self, width=3, length=10):
        r"""
        See equivalent method in :class:`FiniteLattice`. In this case, the
        facets are added to *all* of the finite lattices (one for each symmetry partner).
        """
        for k in range(len(self.lattices)):
            self.lattices[k].make_hexagonal_prism(width=width, length=length, shift=self.au_x_coms[k])

    def make_parallelepiped(self, shape=(5, 5, 5), shift=0):
        r"""
        See equivalent method in :class:`FiniteLattice <reborn.target.crystal.FiniteLattice>`. In this case, the
        facets are added to *all* of the finite lattices (one for each symmetry partner).
        """
        for k in range(len(self.lattices)):
            self.lattices[k].make_parallelepiped(shape=shape, shift=shift)

    def set_gaussian_disorder(self, sigmas=(0, 0, 0)):
        r"""
        See equivalent method in :class:`FiniteLattice` .
        """
        for lat in self.lattices:
            lat.set_gaussian_disorder(sigmas=sigmas)



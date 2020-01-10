.. _working_with_crystals:

Working with Crystals
=====================

Here we briefly outline some conventions assumed in bornagain, and how to use the classes that expose crystal
symmetry operations.

A little bit of math
--------------------

In crystallography applications we frequently work in four different coordinate systems.  We firstly define the usual
*cartesian coordinates* (sometimes called *orthogonal coordinates*), denoted as :math:`\mathbf{r}`.  These are the
usual laboratory coordinates in which the basis vectors are orthogonal and of equal magnitude.  The
*fractional coordinates* are denoted as :math:`\mathbf{x}`.  These coordinates are convenient to use when dealing with
crystallographic symmetry operations (for example).  The basis vectors are related by the
*orthogonalization matrix* :math:`\mathbf{O}` as follows:

.. math:: \mathbf{r} = \mathbf{O}\mathbf{x}

The inverse of the orthogonalization matrix, :math:`\mathbf{O}^{-1}`, is sometimes called the *deorthogonalization
matrix*.   The columns of the orthogonalization matrix are the basis vectors :math:`\mathbf{a}_1`, :math:`\mathbf{a}_2`,
:math:`\mathbf{a}_3` of the crystal:

.. math:: \mathbf{O} = \begin{bmatrix}  | & |  & | \\ \mathbf{a}_1 &  \mathbf{a}_2 & \mathbf{a}_3 \\ | & | & | \end{bmatrix}

In reciprocal space, we have analogous mathematics for the *reciprocal coordinates* :math:`\mathbf{g} = \mathbf{q}/2\pi`
and *fractional Miller indices* :math:`\mathbf{h}`.  They are related by the matrix
:math:`\mathbf{A} = (\mathbf{O}^{-1})^{T}`:

.. math:: \mathbf{g} = \mathbf{A} \mathbf{h}

which contains the reciprocal lattice vectors in its columns:

.. math:: \mathbf{A} = \begin{bmatrix}  | & |  & | \\ \mathbf{a}^*_1 &  \mathbf{a}^*_2 & \mathbf{a}^*_3 \\ | & | & | \end{bmatrix}

The reciprocal lattice vectors are defined as

.. math::

    \mathbf{a}_1^* &= \mathbf{a}_2\times \mathbf{a}_3 / V_c \\
    \mathbf{a}_2^* &= \mathbf{a}_3\times \mathbf{a}_1  / V_c \\
    \mathbf{a}_3^* &= \mathbf{a}_1\times \mathbf{a}_2  / V_c

where :math:`V_c = \mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a}_3)` is the volume of the unit cell.

Oftentimes a crystallographic unit cell is specified in terms of three lattice constants (:math:`a`, :math:`b`,
:math:`c`) and three angles (:math:`\alpha`, :math:`\beta`, :math:`\gamma`).  This of course leads to some ambiguity
since there are only six parameters; the three orientational parameters are missing.  The "standard" way to convert to
the orthogonalization matrix appears to be in appendix A of
`this <https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf>`_ document.

We define the Fourier transform in orthogonal coordinates as

.. math:: F(\mathbf{g}) = \int f(\mathbf{r}) \exp(-i 2 \pi \mathbf{g}^T \mathbf{r}) d^3r

The inverse Fourier transform is

.. math:: f(\mathbf{r}) = \int F(\mathbf{g}) \exp(i 2 \pi \mathbf{g}^T \mathbf{r}) d^3q

In noting the relation :math:`\mathbf{g}^T \mathbf{r} = \mathbf{h}^T \mathbf{x}` we may also define the Fourier
transform in the fractional coordinate basis:

.. math:: F(\mathbf{h}) = V_c \int f(\mathbf{x}) \exp(-i 2 \pi \mathbf{h}^T \mathbf{x}) d^3x

The factor of :math:`V_c` in the above is due to the Jacobian determinant :math:`| \mathbf{O} |`.


Loading a PDB file
------------------

PDB (Protein Data Bank) files are text files that contain information about molecular structures, in particular the
positions and types of all the atoms that make up one or more molecular models, along with symmetry operations, and
more.  PDB files can be downloaded from the `PDB website <http://www.rcsb.org>`_ or using the
:func:`get_pdb_file()<bornagain.target.crystal.get_pdb_file>` function in bornagain.

PDB files have a well-defined `specification <http://www.wwpdb.org/documentation/file-format>`_ and may be divided into
various "records".  Some of the important ones are:

0) `MODEL <http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#MODEL>`_,
   which distinguishes different atomic models.
1) `ATOM <http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM>`_ and
   `HETATM <http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#HETATM>`_, which contain
   orthogonal coordinates of atomic models.
2) `CRYST1 <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#CRYST1>`_, which contains
   the unit cell lattice constants and angles (:math:`a`, :math:`b`, :math:`c`, :math:`\alpha`, :math:`\beta`,
   :math:`\gamma`), the full International Tables for Crystallography’s Hermann-Mauguin symbol, and the Z value (number of polymeric chains
   in a unit cell).
3) `SCALE <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#SCALEn>`_, which contains
   *"transformation from the orthogonal coordinates as contained in the PDB entry to fractional crystallographic
   coordinates"*.  Not that this comprises both a rotation and a translation -- what is the purpose of the translation?
4) `MTRIX <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#MTRIXn>`_, which contains
   *"transformations expressing non-crystallographic symmetry... [that] operate on the coordinates in the entry to yield
   equivalent representations of the molecule in the same coordinate frame"*.  I remain puzzled by this comment: *"If
   coordinates for the representations which are approximately related by the given transformation are present in the
   file, the last “iGiven” field is set to 1"* -- does this mean that we should ignore entries for which iGiven=1?
5) `REMARK 290 <https://www.wwpdb.org/documentation/file-format-content/format32/remarks1.html#REMARK%20290>`_, which
   has the crystallographic symmetry operations.  This entry seems reasonably clear though it is not documented well.
6) `ORIGX <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#ORIGXn>`_, which contains *"the
   transformation from the orthogonal coordinates contained in the entry to the submitted coordinates"*.  We are
   not interested in the original coordinates; we only care about the coordinates in the file.

Based on the above, and on appendix A of
`this <https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf>`_, we present our understanding of
how we should interpret the coordinates and transformations found in a PDB file.

We begin with the orthogonal coordinates :math:`\mathbf{r}_0` of a model that are contained in a PDB file.  If there are
non-crystallographic symmetry (NCS) operations, for example in the case of a virus capsid, we *first* generate the
the complete set of atomic coordinates that comprise the asymmetric unit (AU).  To do this, we generate the NCS
symmetry partners using the matrices :math:`\mathbf{M}_i` and translation vectors
:math:`\mathbf{V}_i` found in the MTRIX record as follows:

.. math:: \mathbf{r}_\text{ncs, i} = \mathbf{M}_i \mathbf{r}_0 + \mathbf{V}_i

From the documentation, there are some entries in the list of :math:`\mathbf{M}`, :math:`\mathbf{V}` that are only
"approximate" symmetries, which I gather are provided "just FYI", and which should *not* be applied to
:math:`\mathbf{r}_0` because the symmetry-related coordinates already appear explicitly in the stored
:math:`\mathbf{r}_0`.

After we do the above we build the crystal asymmetric unit (AU) by concatenating all of the above coordinates to form
:math:`\mathbf{r}_\text{au} = \{\mathbf{r}_\text{ncs}\}`.  In order to generate the crystallographic symmetry partners,
we could use the rotation matrices :math:`\mathbf{R}_n` and translation vectors :math:`\mathbf{T}_n` found in the
REMARK 290 record.  We may apply them to the AU orthogonal coordinates as follows:

.. math:: \mathbf{r}_n = \mathbf{R}_n \mathbf{r}_\text{au} + \mathbf{T}_n
    :label: stupidTrans

Finally, we may transform to fractional coordinates via the matrix :math:`\mathbf{S}` and translation vector
:math:`\mathbf{U}` found in the SCALE record:

.. math:: \mathbf{x} = \mathbf{S} \mathbf{r} + \mathbf{U}
    :label: stupidU

All of the above quantities can be loaded using the :func:`pdb_to_dict()<bornagain.target.crystal.pdb_to_dict()>`
function, which returns a Python dictionary with the following mappings to the notation above:

========================= =========================== ================================================================================
Dictionary key            Data type                   Mathematical symbol
========================= =========================== ================================================================================
'scale_matrix'            Shape (3, 3) array          :math:`\mathbf{S}`
'scale_translation'       Shape (3) array             :math:`\mathbf{U}`
'atomic_coordinates'      Shape (N, 3) array          :math:`\mathbf{r}_0`
'atomic_symbols'          List of strings             e.g. "H", "He", "Li", etc.
'unit_cell'               Length 6 tuple              (:math:`a`, :math:`b`, :math:`c`, :math:`\alpha`, :math:`\beta`, :math:`\gamma`)
'spacegroup_symbol'       String                      e.g. "P 63"
'spacegroup_rotations'    List of shape (3, 3) arrays :math:`\mathbf{R}_n`
'spacegroup_translations' List of shape (3) arrays    :math:`\mathbf{T}_n`
'ncs_rotations'           List of shape (3, 3) arrays :math:`\mathbf{M}_i`
'ncs_translations'        List of shape (3) arrays    :math:`\mathbf{V}_i`
'i_given'                 Shape (M) array of integers N/A
========================= =========================== ================================================================================

Note that the units are not modified from PDB format; angles are degrees and distances are in Angstrom units.  This is
one of the rare cases in which non-SI units are used in bornagain (but we convert to SI immediately when we create a
class from this dictionary).


Crystallographic symmetry operations
------------------------------------

When concerned with crystals, it usually makes sense to work primarily in the fractional coordinates
:math:`\mathbf{x}`.  We wish to have simple crystallographic symmetry operations according to

.. math:: \mathbf{x}_n = \mathbf{W}_n \mathbf{x}_\text{au} + \mathbf{Z}_n

We also wish to have a simple way to move to the orthogonal coordinate system according to

.. math:: \mathbf{r} = \mathbf{O}\mathbf{x}

The benefit of working in the :math:`\mathbf{x}` coordinates in the above way is that the "rotations"
:math:`\mathbf{W}_n` are strictly permutation operators comprised of elements with values -1, 0, 1, and the translations
:math:`\mathbf{Z}_n` are strictly integer multiples of 1/6 or 1/4.  As a result, we can define a mesh of density samples
in which crystallographic operations do not result in interpolations.

We first consider the case in which :math:`\mathbf{U}=0`.  Suppose we have the following from the PDB file:

.. math::

    \mathbf{r}_n &= \mathbf{R}_n \mathbf{r}_\text{au} + \mathbf{T}_n \\
    \mathbf{x} &= \mathbf{S} \mathbf{r}

From the second line we see that :math:`\mathbf{O}=\mathbf{S}^{-1}`.  We do two manipulations of the above equations to
get

.. math::

    \mathbf{S} \mathbf{r}_n &= \mathbf{S} \mathbf{R}_n \mathbf{r}_\text{au} + \mathbf{S} \mathbf{T}_n \\
    \mathbf{x}_n &= \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1}\mathbf{x}_\text{au} + \mathbf{S} \mathbf{T}_n

which gives us our desired transformations:

.. math::

    \mathbf{O} &= \mathbf{S}^{-1} \\
    \mathbf{W}_n &= \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1} \\
    \mathbf{Z}_n &= \mathbf{S}\mathbf{T}_n

Assuming :math:`\mathbf{U}=0`, the :func:`CrystalStructure() <bornagain.target.crystal.CrystalStructure()>` class can be
used to easily load in a PDB file and get :math:`\mathbf{x}_\text{au}` and the transformations :math:`\mathbf{W}_n`, :math:`\mathbf{Z}_n`.

In the uncommon situation where :math:`\mathbf{U} \ne 0`, we do not have an understanding of how to determine the
:math:`\mathbf{x}_\text{au}` and transformations :math:`\mathbf{W}_n`, :math:`\mathbf{Z}_n`.  You will get a warning,
and our best guess as to what the transformations are.  See the Appendix below for more information.


Putting it all together
-----------------------

As an example, the following script will use a PDB file to produce the
coordinates :math:`\mathbf{x}_\text{au}` and transformations :math:`\mathbf{W}_n`, :math:`\mathbf{Z}_n`, and then use
them to generate the second crystallographic symmetry partner :math:`\mathbf{x}_2`:

.. code-block:: python

    import numpy as np
    from bornagain.data import lysozyme_pdb_file
    from bornagain.target import crystal
    cryst = crystal.CrystalStructure(lysozyme_pdb_file)
    x_au = cryst.fractional_coordinates
    W2 = cryst.spacegroup.sym_rotations[1]
    Z2 = cryst.spacegroup.sym_translations[1]
    x2 = np.dot(x_au, W2.T) + Z2

We could go on to get other quantities such as atomic scattering factors:

.. code-block:: python

    import scipy
    eV = scipy.constants.value('electron volt')
    photon_energy = 9500 * eV
    f = cryst.molecule.get_scattering_factors(photon_energy)


Appendix
--------

**PDB transformation confusion**

We have a problem if :math:`\mathbf{U} \ne 0`.  Combining :eq:`stupidU` and :eq:`stupidTrans` and performing a few
manipulations gives

.. math::

    \mathbf{x}_n = \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1} \mathbf{x}_\text{au}  + \mathbf{S}\mathbf{T}_n + (\mathbf{I} - \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1})\mathbf{U}

or, equivalently,

.. math::

    \mathbf{x}_n = \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1} (\mathbf{x}_\text{au} - \mathbf{U})  + \mathbf{S}\mathbf{T}_n + \mathbf{U}

The transformations we desire are now ambiguous.  One option is to re-define
:math:`\mathbf{x}_\text{au} - \mathbf{U} \rightarrow \mathbf{x}_\text{au}` and choose the translation
:math:`\mathbf{Z}_n = \mathbf{S}\mathbf{T}_n + \mathbf{U}`.  A second option is to leave :math:`\mathbf{x}_\text{au}`
alone, but then we have a different expression for :math:`\mathbf{Z}_n`.  The correct answer should ensure that
:math:`\mathbf{Z}_n` is composed of integer multiples of 1/6 or 1/4.  The strange thing is that we get the correct
operations only if we set :math:`\mathbf{U} = 0`.  This can be seen for example in the case of the PDB file 1lsp.pdb.
Look to the test file ``test_pdb.py`` for more details.


.. .. math::

        \mathbf{O} = \mathbf{S}^{-1}

        \mathbf{W}_n = \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1}

        \mathbf{Z}_n = \mathbf{S}\mathbf{T}_n + (\mathbf{I} - \mathbf{W}_n)\mathbf{U}

    Another option is to re-define the asymmetric unit and then define

    .. math::

        \mathbf{x}_\text{au} \leftarrow \mathbf{x}_\text{au} - \mathbf{U}

        \mathbf{O} = \mathbf{S}^{-1}

        \mathbf{W}_n = \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1}

        \mathbf{Z}_n = \mathbf{S}\mathbf{T}_n + \mathbf{U}

    Which of the above is correct?  So far, our tests have not yielded a clear answer.  We want to ensure that
    :math:`\mathbf{Z}_n` is composed of integer multiples of 1/6 or 1/4.
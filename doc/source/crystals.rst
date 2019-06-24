Working with crystals
=====================

Here we briefly outline some conventions assumed in bornagain, and how to use the classes that expose crystal
symmetry operations.

A little bit of math
--------------------

We define the *orthogonal coordinates*, denoted as :math:`\mathbf{r}`.  These are the usual "Cartesian" laboratory
coordinates.  The *fractional coordinates* are denoted
as :math:`\mathbf{x}`.  These two vectors are related by the *orthogonalization matrix* :math:`\mathbf{O}` as follows:

.. math:: \mathbf{r} = \mathbf{O}\mathbf{x}

The inverse of the orthogonalization matrix, :math:`\mathbf{O}^{-1}`, is sometimes called the "deorthogonalization
matrix".   The columns of the orthogonalization matrix are the basis vectors :math:`\mathbf{a}_1`, :math:`\mathbf{a}_2`,
:math:`\mathbf{a}_3` of the crystal:

.. math:: \mathbf{O} = \begin{bmatrix}  | & |  & | \\ \mathbf{a}_1 &  \mathbf{a}_2 & \mathbf{a}_3 \\ | & | & | \end{bmatrix}

In reciprocal space, we have analogous mathematics for the *reciprocal coordinates* :math:`\mathbf{q}` and *fractional
Miller indices* :math:`\mathbf{h}`.  They are related by the matrix :math:`\mathbf{A} = (\mathbf{O}^{-1})^{T}`:

.. math:: \mathbf{q} = \mathbf{A} \mathbf{h}

which contains the reciprocal lattice vectors in its columns:

.. math:: \mathbf{A} = \begin{bmatrix}  | & |  & | \\ \mathbf{a}^*_1 &  \mathbf{a}^*_2 & \mathbf{a}^*_3 \\ | & | & | \end{bmatrix}

The reciprocal lattice vectors are defined as

.. math::

    \mathbf{a}_1^* = \mathbf{a}_2\times \mathbf{a}_3 / V_c

    \mathbf{a}_2^* = \mathbf{a}_3\times \mathbf{a}_1  / V_c

    \mathbf{a}_3^* = \mathbf{a}_1\times \mathbf{a}_2  / V_c

where :math:`V_c = \mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a}_3)` is the volume of the unit cell.

Often times a crystallographic unit cell is specified in terms of three lattice constants (:math:`a`, :math:`b`,
:math:`c`) and three angles (:math:`\alpha`, :math:`\beta`, :math:`\gamma`).  This of course leads to some ambiguity
since there are only six parameters; the three orientational parameters are missing.  Te "standard" way to convert to
the orthogonalization matrix appears to be in the appendix of
`this <https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf>`_ document, for example.



We define the Fourier transform in orthogonal coordinates as

.. math:: F(\mathbf{q}) = \int f(\mathbf{r}) \exp(-i 2 \pi \mathbf{q}^T \mathbf{r}) d^3r

Note that there is no factor of :math:`2\pi` in the definition of :math:`\mathbf{q}` in this section.  The inverse
Fourier transform is

.. math:: f(\mathbf{r}) =\frac{1}{(2\pi)^3}\int F(\mathbf{q}) \exp(i 2 \pi \mathbf{q}^T \mathbf{r}) d^3q

In noting the relation :math:`\mathbf{q}^T \mathbf{r} = \mathbf{h}^T \mathbf{x}` we may also define the Fourier
transform in the fractional coordinate basis:

.. math:: F(\mathbf{h}) = V_c \int f(\mathbf{x}) \exp(-i 2 \pi \mathbf{h}^T \mathbf{x}) d^3x

The factor of :math:`V_c` in the above is due to the Jacobian determinant :math:`| \mathbf{O} |`.


Loading (and understanding) a PDB file
--------------------------------------

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
   :math:`\gamma`), the full International Table’s Hermann-Mauguin symbol, and the Z value (number of polymeric chains
   in a unit cell).
3) `SCALE <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#SCALEn>`_, which contains
   "transformation from the orthogonal coordinates as contained in the PDB entry to fractional crystallographic
   coordinates".  Not that this comprises both a rotation and a translation -- what is the purpose of the translation?
4) `MTRIX <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#MTRIXn>`_, which contains
   "transformations expressing non-crystallographic symmetry... [that] operate on the coordinates in the entry to yield
   equivalent representations of the molecule in the same coordinate frame".  I remain puzzled by this comment: "If
   coordinates for the representations which are approximately related by the given transformation are present in the
   file, the last “iGiven” field is set to 1" -- does this mean that we should ignore entries for which iGiven=1?
5) `REMARK 290 <https://www.wwpdb.org/documentation/file-format-content/format32/remarks1.html#REMARK%20290>`_, which
   has the crystallographic symmetry operations.  This entry seems reasonably clear though it is not documented well.
6) `ORIGX <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#ORIGXn>`_, which contains "the
   transformation from the orthogonal coordinates contained in the entry to the submitted coordinates".  We are
   not interested in the original coordinates; we only care about the coordinates in the file.

Given the above, here is my present understanding of how we should interpret the entries in a PDB file:

The orthogonal coordinates :math:`\mathbf{r}` of a model contained in a PDB file are likely only a subset of the
coordinates that you need in your project.  If there are non-crystallographic symmetry (NCS) operations, for example
in the case of a virus
capsid, you should *first* generate the NCS symmetry partners using the matrices :math:`\mathbf{R}_\text{ncs}` and
translation vectors :math:`\mathbf{T}_\text{ncs}` found in the MTRIX record as follows:

.. math:: \mathbf{r}_\text{ncs} = \mathbf{R}_\text{ncs} \mathbf{r} + \mathbf{T}_\text{ncs}

After you do the above you have all the atomic coordinates that comprise the crystal asymmetric unit (AU).  We
concatenate all of these coordinates to form the coordinates of the AU, denoted as :math:`\mathbf{r}_\text{au}`.
In order to
generate the crystallographic symmetry partners, you can use the matrices :math:`\mathbf{R}_n` and translation vectors
:math:`\mathbf{T}_n` found in the REMARK 290 record.  Apply the following to the AU orthogonal coordinates:

.. math:: \mathbf{r}_n = \mathbf{R}_n \mathbf{r}_\text{au} + \mathbf{T}_n
    :label: stupidTrans

Finally, we may transform to fractional coordinates via the matrix :math:`\mathbf{S}` and translation vector
:math:`\mathbf{U}` found in the SCALE record:

.. math:: \mathbf{x}_n = \mathbf{S} \mathbf{r}_n + \mathbf{U}
    :label: stupidU

All of the above quantities can be loaded using the
:func:`pdb_to_dict()<bornagain.target.crystal.pdb_to_dict()>` function, which returns a Python dictionary with the
following mappings to the notation above:

========================= =========================== ================================================================================
Dictionary key            Data type                   Mathematical symbol
========================= =========================== ================================================================================
'scale_matrix'            Shape (3, 3) array          :math:`\mathbf{S}`
'scale_translation'       Shape (3) array             :math:`\mathbf{U}`
'atomic_coordinates'      Shape (N, 3) array          :math:`\mathbf{r}`
'atomic_symbols'          List of strings             e.g. "H", "He", "Li", etc.
'unit_cell'               Length 6 tuple              (:math:`a`, :math:`b`, :math:`c`, :math:`\alpha`, :math:`\beta`, :math:`\gamma`)
'spacegroup_symbol'       String                      e.g. "P 63"
'spacegroup_rotations'    List of shape (3, 3) arrays :math:`\mathbf{R}_n`
'spacegroup_translations' List of shape (3) arrays    :math:`\mathbf{T}_n`
'ncs_rotations'           List of shape (3, 3) arrays :math:`\mathbf{R}_\text{ncs}`
'ncs_translations'        List of shape (3) arrays    :math:`\mathbf{T}_\text{ncs}`
========================= =========================== ================================================================================

Note that the units are not modified from PDB format; angles are degrees and distances are in Angstrom units.


Crystallographic symmetry operations
------------------------------------

When concerned with crystals, it usually makes sense to work primarily in the fractional coordinates
:math:`\mathbf{x}`.  We wish to have simple crystallographic symmetry operations according to

.. math:: \mathbf{x}_n = \mathbf{W}_n \mathbf{x}_\text{au} + \mathbf{Z}_n

We also wish to have a simple way to move to the orthogonal coordinate system according to

.. math:: \mathbf{r} = \mathbf{O}\mathbf{x}

The benefit of working in the :math:`\mathbf{x}` coordinates in the above way is that the "rotations"
:math:`\mathbf{W}_n` are strictly permutation operators comprised of elements with values -1, 0, 1, and the translations
:math:`\mathbf{Z}_n` are strictly integer multiples of 1/6 or 1/4.
As a result, we can define a mesh of density samples in which crystallographic operations
do not result in interpolations.

Combining :eq:`stupidU` and :eq:`stupidTrans`, we see that the PDB specification does not provide such a simple mapping.
Symmetry-related fractional coordinates are determined by the following operation:

.. math::

    \mathbf{x}_n = \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1} \mathbf{x}_\text{au}  + \mathbf{S}\mathbf{T}_n + (\mathbf{I} - \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1})\mathbf{U}

or, equivalently,

.. math::

    \mathbf{x}_n = \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1} (\mathbf{x}_\text{au} - \mathbf{U})  + \mathbf{S}\mathbf{T}_n + \mathbf{U}

Now we see that the transformations we desire, in terms of what we get from a PDB file, are either

.. math::

    \mathbf{O} = \mathbf{S}^{-1}

    \mathbf{W}_n = \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1}

    \mathbf{Z}_n = \mathbf{S}\mathbf{T}_n + (\mathbf{I} - \mathbf{W}_n)\mathbf{U}

or we can re-define the asymmetric unit first and define

.. math::

    \mathbf{x}_\text{au} \leftarrow \mathbf{x}_\text{au} - \mathbf{U}

    \mathbf{O} = \mathbf{S}^{-1}

    \mathbf{W}_n = \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1}

    \mathbf{Z}_n = \mathbf{S}\mathbf{T}_n + \mathbf{U}

Which of the above is correct?  We want to ensure that :math:`\mathbf{Z}_n` is composed of integer multiples of 1/6 or
1/4.

The :func:`CrystalStructure() <bornagain.target.crystal.CrystalStructure()>` class can be used to easily load in a PDB
file and generate symmetry partners.  For example, the following script will produce the coordinates
:math:`\mathbf{x}_\text{au}` and transformations :math:`\mathbf{W}_n`, :math:`\mathbf{Z}_n`, and then use them to
generate the second crystallographic symmetry partner :math:`\mathbf{x}_2`:

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









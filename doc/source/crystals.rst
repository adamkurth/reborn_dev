Working with crystals
=====================

Here we briefly outline some conventions assumed in bornagain, and how to use the classes that expose crystal
symmetry operations.

A little bit of math
--------------------

We define the *orthogonal coordinates*, denoted as :math:`\mathbf{r}`.  These are the usual "Cartesian" laboratory
coordinates.  The *fractional coordinates* are denoted
as :math:`\mathbf{x}`.  These two vectors are related by the *orthogonalization matrix* :math:`\mathbf{O}` as follows:

.. math::\mathbf{r} = \mathbf{O}\mathbf{x}

The inverse of the orthogonalization matrix, :math:`\mathbf{O}^{-1}`, is sometimes called the "deorthogonalization
matrix"

The columns of the orthogonalization matrix are the basis vectors :math:`\mathbf{a}_1`, :math:`\mathbf{a}_2`,
:math:`\mathbf{a}_3` of the crystal:

.. math::\mathbf{O} = \begin{bmatrix}  | & |  & | \\ \mathbf{a}_1 &  \mathbf{a}_2 & \mathbf{a}_3 \\ | & | & | \end{bmatrix}

In reciprocal space, we have analogous mathematics for the *reciprocal coordinates* :math:`\mathbf{q}` and *fractional
Miller indices* :math:`\mathbf{h}`.  They are related by the :math:`\mathbf{A}` matrix:

.. math::\mathbf{q} = \mathbf{A} \mathbf{h}

which contains the reciprocal lattice vectors in its columns:

.. math::\mathbf{A} = \begin{bmatrix}  | & |  & | \\ \mathbf{a}^*_1 &  \mathbf{a}^*_2 & \mathbf{a}^*_3 \\ | & | & | \end{bmatrix}

The reciprocal lattice vectors are defined as

.. math::

    \mathbf{a}_1^* = \mathbf{a}_2\times \mathbf{a}_3 / V_c

    \mathbf{a}_2^* = \mathbf{a}_3\times \mathbf{a}_1  / V_c

    \mathbf{a}_3^* = \mathbf{a}_1\times \mathbf{a}_2  / V_c

where :math:`V_c = \mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a}_3)` is the volume of the unit cell.

It is useful to know the following relations:

.. math::

    \mathbf{A} = (\mathbf{O}^{-1})^{T}

    \mathbf{q}^T \mathbf{r} = \mathbf{h}^T \mathbf{x}

We define the Fourier transform in orthogonal coordinates as

.. math::F(\mathbf{q}) = \int f(\mathbf{r}) \exp(-i 2 \pi \mathbf{q}^T \mathbf{r}) d^3r

Note that there is no factor of :math:`2\pi` in the definition of :math:`\mathbf{q}` in this section.  The inverse
Fourier transform is

.. math::f(\mathbf{r}) =\frac{1}{(2\pi)^3}\int F(\mathbf{q}) \exp(i 2 \pi \mathbf{q}^T \mathbf{r}) d^3q

We may also define the Fourier transform in the fractional coordinate basis:

.. math::F(\mathbf{h}) = V_c \int f(\mathbf{x}) \exp(-i 2 \pi \mathbf{h}^T \mathbf{x}) d^3x

The factor of :math:`V_c` in the above is due to the Jacobian determinant :math:`| \mathbf{O} |`.


Loading (and understanding) a PDB file
--------------------------------------

PDB (Protein Data Bank) files are text files that contain information about molecular structures, in particular the
positions and types of all the atoms that make up the molecules, along with symmetry operations, and much more.
PDB files can be downloaded from the `PDB website <http://www.rcsb.org>`_ or using the
:func:`get_pdb_file()<bornagain.target.crystal.get_pdb_file>` function in bornagain.

PDB files have a well-defined `specification <http://www.wwpdb.org/documentation/file-format>`_ and may be divided into
various "records".  Some of the important ones are:

0) MODEL, which distinguishes different atomic models,
1) `ATOM <http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM>`_ and
   `HETATM <http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#HETATM>`_, which contain
   orthogonal coordinates of atomic models,
2) `CRYST1 <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#CRYST1>`_, which contains
   the unit cell and a spacegroup symbol,
3) `SCALE <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#SCALEn>`_, which contains
   transformations that convert orthogonal coordinates to fractional coordinates,
4) `MTRIX <http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#MTRIXn>`_, which contains
   transformations for non-crystallographic symmetry,
5) `REMARK 290 <https://www.wwpdb.org/documentation/file-format-content/format32/remarks1.html#REMARK%20290>`_, which
   has the crystallographic symmetry operations.

The orthogonal coordinates :math:`\mathbf{r}` of a model in a PDB file are likely only a subset of the coordinates
that you need in your project.  If there is non-crystallographic symmetry (NCS), for example in the case of a virus
capsid, you should first generate the NCS symmetry partners using the matrices :math:`\mathbf{R}_\text{ncs}` and
translation vectors :math:`\mathbf{T}_\text{ncs}` found in the MTRIX record as follows:

.. math::\mathbf{r}_\text{ncs} = \mathbf{R}_\text{ncs} \mathbf{r} + \mathbf{T}_\text{ncs}

After you do the above you have all the atomic coordinates that comprise the crystal asymmetric unit (AU).  We
concatenate all of these coordinates to form the coordinates of the AU, denoted as :math:`\mathbf{r}_\text{au}`.
In order to
generate the crystallographic symmetry parnters, you can use the matrices :math:`\mathbf{R}_n` and translation vectors
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
:math:`\mathbf{x}` .  We wish to have simple crystallographic symmetry operations according to

.. math:: \mathbf{x}_n = \mathbf{W}_n \mathbf{x}_\text{au} + \mathbf{Z}_n

We also wish to have a simple way to move to the orthogonal coordinate system according to

.. math:: \mathbf{r} = \mathbf{O}\mathbf{x}

The benefit of working in the :math:`\mathbf{x}` coordinates in the above way is that the "rotations"
:math:`\mathbf{W}_n` are strictly permutation operators comprised of elements with values -1, 0, 1, and the translations
:math:`\mathbf{Z}_n` are strictly integer multiples of 1/6 or 1/4.
As a result, we can define a mesh of density samples in which crystallographic operations
do not result in interpolations.

Combining :eq:`stupidU` and :eq:`stupidTrans` we have

.. math::

    \mathbf{x}_n = \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1} \mathbf{x}_\text{au}  + \mathbf{S}\mathbf{T}_n + (\mathbf{I} - \mathbf{S} \mathbf{R}_n \mathbf{S}^{-1})\mathbf{U}

Now we see that the transformations we desire, in terms of what we get from a PDB file, are

.. math::

    \mathbf{O} = \mathbf{S}^{-1}

    \mathbf{W}_n = \mathbf{S} \mathbf{R}_n' \mathbf{S}^{-1}

    \mathbf{Z}_n = \mathbf{S}\mathbf{T}_n + (\mathbf{I} - \mathbf{W}_n)\mathbf{U}






To load a PDB file using bornagain, use :func:`CrystalStructure()<bornagain.target.crystal.CrystalStructure()>` and pass in the PDB file, something like:

.. code-block:: python

	import bornagain.target.crystal as crystal
	prot = crystal.CrystalStructure(pdbFile)

this will return an object called prot which has methods for making a density map etc.


Getting scattering factors
---------------------------
The scattering factors are the Fourier transform. They are complex numbers. You can get the scattering factors of a molecule by 

.. code-block:: python

	f = bornagain.simulate.atoms.get_scattering_factors(prot.Z, bornagain.units.hc / wavelength)









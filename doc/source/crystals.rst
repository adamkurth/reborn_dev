Working with crystals
=====================

A little bit of math
--------------------

Here we briefly outline some conventions assumed in bornagain.

We firstly define the *orthogonal coordinates*, which we denote as :math:`\mathbf{r}`.  These are the usual "Cartesian"
coordinates.  The other coordinate system that is relevant to us is the *fractional coordinates*, which we denote here
as :math:`\mathbf{x}`.  These two vectors are related by the *orthogonalization matrix* :math:`\mathbf{O}` as follows:

.. math ::

    \mathbf{r} = \mathbf{O}\mathbf{x}

The columns of the orthogonalization matrix are the basis vectors :math:`\mathbf{a}_1`, :math:`\mathbf{a}_2`,
:math:`\mathbf{a}_3` of the crystal:

.. math ::

    \mathbf{O} = \begin{bmatrix}  | & |  & | \\ \mathbf{a}_1 &  \mathbf{a}_2 & \mathbf{a}_3 \\ | & | & | \end{bmatrix}

In reciprocal space, we have analogous mathematics for the *reciprocal coordinates* :math:`\mathbf{q}` and *fractional
Miller indices* :math:`\mathbf{h}`.  They are related by the :math:`\mathbf{A}` matrix:

.. math ::

    \mathbf{q} = \mathbf{A} \mathbf{h}

which contains the reciprocal lattice vectors in its columns:

.. math ::

    \mathbf{A} = \begin{bmatrix}  | & |  & | \\ \mathbf{a}^*_1 &  \mathbf{a}^*_2 & \mathbf{a}^*_3 \\ | & | & | \end{bmatrix}

The reciprocal lattice vectors are defined as

.. math ::

    \mathbf{a}_1^* = \mathbf{a}_2\times \mathbf{a}_3 / V_c

    \mathbf{a}_2^* = \mathbf{a}_3\times \mathbf{a}_1  / V_c

    \mathbf{a}_3^* = \mathbf{a}_1\times \mathbf{a}_2  / V_c

where :math:`V_c = \mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a}_3)` is the volume of the unit cell.

It is useful to know the following relations:

.. math ::

    \mathbf{A} = (\mathbf{O}^{-1})^{T}

    \mathbf{q}^T \mathbf{r} = \mathbf{h}^T \mathbf{x}


Loading a PDB file
------------------
PDB (Protein Data Bank) files are text files that contain information about a molecule, in particular, it has the positions and types of all the atoms that make up that molecule.
PDB files can be downloaded from the `PDB website <http://www.rcsb.org>`_ or using the :func:`get_pdb_file()<bornagain.target.crystal.get_pdb_file>` function in bornagain.

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


Symmetry operations
-------------------




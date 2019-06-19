Working with crystals
=====================

A little bit of math
--------------------

Here we briefly outline some conventions assumed in bornagain.

We firstly define the *orthogonal coordinates*, which we denote as :math:`\mathbf{r}`.  These are the usual "Cartesian"
coordinates.  The other coordinate system taht is relevant to us is the *fractional coordinates*, which we denote here
as :math:`\mathbf{x}`.  These two vectors are related by the *orthogonalization matrix* :math:`\mathbf{O}` as follows:

.. math ::

    \mathbf{r} = \mathbf{O}\mathbf{x}

The columns of the orthogonalization matrix are the basis vectors :math:`\mathbf{a}`, :math:`\mathbf{b}`,
:math:`\mathbf{c}` of the crystal:

.. math ::

    \mathbf{O} = \begin{bmatrix}  | & |  & | \\ \mathbf{a} &  \mathbf{b} & \mathbf{c} \\ | & | & | \end{bmatrix}

In reciprocal space, we have analogous mathematics for the *reciprocal coordinates* :math:`\mathbf{q}` and *fractional
Miller indices* :math:`\mathbf{h}`.  They are related by the :math:`\mathbf{A}` matrix:

.. math ::

    \mathbf{q} = \mathbf{A} \mathbf{h}

which contains the reciprocal lattice vectors in its columns:

.. math ::

    \mathbf{A} = \begin{bmatrix}  | & |  & | \\ \mathbf{a}^* &  \mathbf{b}^* & \mathbf{c}^* \\ | & | & | \end{bmatrix}

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





RenormalizeDiag
~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
For a given squre matrix this transformation either:

1. Scales :math:`n` diagonals by a given number.
2. Scales all the elements except :math:`n` diagonals by a given number.

After scale is applied, each column is normalized to 1.

RenormalizeDiag_ is used to implement IAV correction uncertainty (Daya Bay)
and is supposed to be used with EnergySmear transformation.

Inputs
^^^^^^

1. ``'renorm.inmat'`` — :math:`C` square input matrix.

Outputs
^^^^^^^

1. ``'renorm.outmat'`` — :math:`E` square output matrix.

Variables
^^^^^^^^^

1. ``'DiagScale'`` — :math:`s` scale to be applied.

Arguments
^^^^^^^^^

1. ``int ndiag`` — :math:`n` number of diagonals to treat as as diagonal:

   1 — only diagonal itself

   2 — the diagonal itself, upper and lower diagonal

   3 — etc

2. ``Target target`` — ``RenormalizeDiag::Target::Diagonal`` (default) or ``RenormalizeDiag::Target::Offdiagonal``.
   Shows which part of the matrix to apply the scale to.

2. ``Mode mode`` — ``RenormalizeDiag::Mode::Full`` (default) or ``RenormalizeDiag::Mode::Upper``.
   If ``Upper``, the matrix is treated as upper triangular matrix with appropriate optimizations.

3. ``parname='DiagScale'`` — variable name.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/detector/test_renormalizediag.py
   ./tests/detector/test_iavunc.py -s

Implementation
^^^^^^^^^^^^^^

The result is matrix, normalized by sum of rows:

.. math::
   E_{ij} = \frac{D_{ij}}{\sum\limits_k D_{kj}},

where :math:`D` in ``Diagonal`` case is:

.. math::
   D_{ij} =
    \begin{cases}
     s C_{ij} &            \text{if } |i-j|<n \\
     \phantom{s} C_{ij}   & \text{otherwise}
    \end{cases},

and in ``Offdiagonal`` case:

.. math::
   D_{ij} =
    \begin{cases}
     \phantom{s} C_{ij} &            \text{if } |i-j|<n \\
     s C_{ij}           & \text{otherwise}
    \end{cases}.

The ``Upper`` mode ensures that the lower triangle (without diagonal) is set to zero *before* normalization:

.. math::
   D_{ij} = 0\text{ if } i>j.

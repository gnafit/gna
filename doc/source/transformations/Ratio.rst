.. _Ratio:

Ratio
~~~~~

Description
^^^^^^^^^^^

Calculates an element-wise ratio of two arrays or matrices.

Inputs
^^^^^^

1) ``ratio.top`` — the nominator: array or matrix :math:`A`.

2) ``ratio.bottom`` — the denominator: array or matrix :math:`B`.

All the inputs have to be of the same dimension.

Outputs
^^^^^^^

1) ``'ratio.ratio'`` — the ratio :math:`R`.

Implementation
^^^^^^^^^^^^^^

Computes a ratio :math:`R`:

.. math::
   R_{ij} = \frac{A_{ij}}{B_{ij}}

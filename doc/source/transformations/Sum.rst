.. _Sum:

Sum
~~~

Description
^^^^^^^^^^^

Calculates a sum of arrays or matrices.

Inputs
^^^^^^

1) Array or matrix :math:`a_1`.

2) Array or matrix :math:`a_2`.

N) Array or matrix :math:`a_N`.

The inputs are passed via ``add()`` method.

All the inputs have to be of the same dimension.

Outputs
^^^^^^^

1) ``'sum.sum'`` — the sum :math:`S`.

Implementation
^^^^^^^^^^^^^^

Computes a sum :math:`S`:

.. math::
   S = \sum\limits_{k=1}^N a_k.

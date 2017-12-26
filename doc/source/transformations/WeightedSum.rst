.. _WeightedSum:

WeightedSum
~~~~~~~~~~~

Description
^^^^^^^^^^^
Calcualate weighted sum of arrays.

Arguments
^^^^^^^^^
1) ``labels`` — vector of arrays' names of size :math:`N`.
2) ``weight_labels`` — vector of weights' names, empty or of size :math:`N`.

Inputs
^^^^^^

1) ``sum.name1`` — Array :math:`A_1`.

2) ``sum.name2`` — :math:`A_2`.

N) ``sum.nameN`` — Array :math:`A_N`.

All arrays are of the same size.

Variables
^^^^^^^^^

The transformation depends on :math:`N` weighting variables :math:`\omega_i`.
Variable names are read from ``weight_labels`` if it is not empty. Otherwise weight labels are defined as
`"weight_"+label[i]`.

Outputs
^^^^^^^

1) ``'sum'`` — weighted sum :math:`S`.

Implementation
^^^^^^^^^^^^^^

The transformation implements a sum

.. math::
    S = \sum_{i=1}^{N} \omega_i A_i,

where weights :math:`\omega_i` are predefined variables.


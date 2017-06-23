Weighted sum
~~~~~~~~~~~~

Description
^^^^^^^^^^^
Calcualate weighted sum of arrays

Constructor arguments
^^^^^^^^^^^^^^^^^^^^^
1) Vector of arrays' names of size :math:`N`

Inputs
^^^^^^

1) Array :math:`A_1`

2) Array :math:`A_2`

N) Array :math:`A_N`

All arrays are of the same size.

Variables
^^^^^^^^^

The transformation depends on :math:`N` weighting variables :math:`\omega_i` 
defined with names starting with `weight_` and ending with relevant array's name.

Outputs
^^^^^^^

1) Weighted sum :math:`S`

Implementation
^^^^^^^^^^^^^^

The transformation implements a sum

.. math::
    S = \sum_{i=1}^{N} \omega_i A_i,

where weights :math:`\omega_i` are predefined variables with names `weight_<name_i>`.


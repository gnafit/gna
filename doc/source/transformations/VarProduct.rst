.. _VarProduct:

VarProduct
~~~~~~~~~~

Description
^^^^^^^^^^^
Calculates the product of at least two variables.

Arguments
^^^^^^^^^

1) ``varnames`` — vector with variable names.
2) ``prodname`` — the name of the resulting variable.

Implementation
^^^^^^^^^^^^^^

The transformation reads each variable by name from ``varnames`` and computes the product:

.. math::
   \text{prodname} = \text{varname}_0 \cdot \text{varname}_1 \cdot \dots.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/elementary/test_varproduct.py



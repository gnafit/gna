.. _VarSum:

VarSum
~~~~~~~

Description
^^^^^^^^^^^
Calculates the sum of at least two variables.

Arguments
^^^^^^^^^

1) ``varnames`` — vector with variable names.
2) ``sumname`` — the name of the resulting variable.

Implementation
^^^^^^^^^^^^^^

The transformation reads each variable by name from ``varnames`` and computes the sum:

.. math::
   \text{sumname} = \text{varname}_0 + \text{varname}_1 + \dots.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/elementary/test_varsum.py

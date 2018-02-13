.. _VarDiff:

VarDiff
~~~~~~~

Description
^^^^^^^^^^^
Calculates the difference of at least two variables.

Arguments
^^^^^^^^^

1) ``varnames`` — vector with variable names.
2) ``diffname`` — the name of the resulting variable.

Implementation
^^^^^^^^^^^^^^

The transformation reads each variable by name from ``varnames`` and computes the product:

.. math::
   \text{diffname} = \text{varname}_0 - \text{varname}_1 - \text{varname}_1 - \dots.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/elementary/test_vardiff.py


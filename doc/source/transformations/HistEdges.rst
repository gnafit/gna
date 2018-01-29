HistEdges transformation
~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
For a given histogram returns bin edges as ``Points``.

Inputs
^^^^^^

1. ``histedges.hist`` — The input edges are defined from it.

Outputs
^^^^^^^

1. ``histedges.edges`` — the output array with binning.

The output is frozen after invocation.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/elementary/test_histedges.py


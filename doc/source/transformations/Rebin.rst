Rebin
~~~~~

Description
^^^^^^^^^^^
Performs the histogram rebinning. The rebinning is implemented via multiplication by a sparse matrix.

Inputs
^^^^^^

1. ``rebin.histin`` — the input histogram. The input edges are defined from it.

Outputs
^^^^^^^

1. ``rebin.histout`` — the output histogram with new binning.

Arguments
^^^^^^^^^

1. ``size_t n`` — number of bin edges.
2. ``double* edges`` — bin edges.
3. ``int rounding`` — number of decimal places to round to. Should be sufficient to distinguish nearest bins.


Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/elementary/test_rebinner.py

Implementation
""""""""""""""

The algorithm is the following:

1. On initialization both edge specifications are rounded to the specified precision in order to avoid float comparison
   uncertainty.
2. Bins below ``edges[0]`` are ignored.
3. Relevant conversion matrix items are filled with 1.
4. Bins above ``edges[-1]`` are ignored.

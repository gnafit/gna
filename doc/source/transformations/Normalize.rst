.. _Normalize:

Normalize
~~~~~~~~~

Description
^^^^^^^^^^^
For a given histogram or an array divide each bin/element by a sum of all the elements.

In 1d case the Normalize may also normalize by a sum of a range (subhistogram).

Inputs
^^^^^^

1. ``normalize.inp`` — :math:`A` input array or histogram. Should be 1-dimensional in case of a subrange.

Outputs
^^^^^^^

1. ``normalize.out`` — :math:`B` output array of the same shape as :math:`A`.

Arguments
^^^^^^^^^

In case no arguments passed the input array or histogram is normalized to the hole sum.

1. ``size_t start`` — :math:`s` first bin of the integral.
2. ``size_t length`` — :math:`n` number of bins to integrate starting from :math:`s`.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/detector/test_normalize.py

Implementation
^^^^^^^^^^^^^^

The result is an array/histogram, normalized by a sum:

.. math::
   B_{j,\dotsc} = \frac{A_{j,\dotsc}}{\sum\limits_{i,\dotsc}A_{i,\dotsc}}.

When the integration range is specified the formula is the following:

.. math::
   B_j = \frac{A_j}{\sum\limits_{i=s}^{s+n-1}A_{i}}.

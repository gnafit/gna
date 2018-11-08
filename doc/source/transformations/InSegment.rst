.. _InSegment:

InSegment transformation
~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Return the indices of the segments to which each value in input array belongs. The InSegment transformation is used for
example in exponential interpolator :ref:`InterpExpoU <InterpExpoU>`.

Inputs
^^^^^^

1) ``insegment.points`` — the points to determine the indices for. Array of any shape of total size :math:`M`.
2) ``insegment.edges`` — the edges of bins/segments of size :math:`N`.

Outputs
^^^^^^^

1) ``insegment.insegment`` — array with indices of the same shape as ``insegment.points``.
2) ``insegment.widths`` — the widths of segments of size :math:`N-1`.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/elementary/test_insegment.py

Implementation
^^^^^^^^^^^^^^

For each value :math:`x_i`, finds the index :math:`j` such that:

.. math::
   e_j <= x_i < e_{j+1},

where :math:`e` is an array of bin edges.

Special cases:

    1) :math:`j=-1` when :math:`x_i<e_0`.
    2) :math:`j=N-1` when :math:`x_i>=e_N`, where :math:`N` is the number of edges.
    3) In case :math:`x_i` is determined to be in bin :math:`j` but close to the next bin :math:`j+1` edge such that
       :math:`e_{j+1}-x_i<T` it is reassigned to bin :math:`j+1`. Here :math:`T` is tolerance with default value of
       :math:`T=10^{-16}`.



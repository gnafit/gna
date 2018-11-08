.. _InterpExpoU:

InterpExpoU -- exponential interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
For a function given as :math:`x,y` computes the result of exponential interpolation (linear vs :math:`log(y)`) for a
given vector of :math:`x'`. The values in :math:`x'` may be unsorted.

The class inherits from :ref:`InSegment` and thus contains two transformations: ``insegment``, ``interp``.

Inputs
^^^^^^

1) ``interp.newx`` — array of :math:`x'` of any shape :math:`S` of points for interpolation.
2) ``interp.x`` — the array :math:`x` of size :math:`N`.
3) ``interp.y`` — the array :math:`y` of size :math:`N` or :math:`y=f(x)` to interpolate.
4) ``interp.segments`` — array of shape :math:`S` with indices assigning each of :math:`x'` to intervals of :math:`x`.
5) ``interp.widths`` — widths of a segments of :math:`x`.

Segments and widths are naturally given by :ref:`InSegment` transformation. When the inputs are connected via
``interpolate(x, y, newx)`` method the :ref:`InSegment` transformation is connected automatically.

Outputs
^^^^^^^

1) ``interp.interp`` — the result of the interpolation of shape :math:`S`.
 
Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/elementary/InterpExpoU.py

Implementation
^^^^^^^^^^^^^^

For a given vectors :math:`x`, :math:`y` and array :math:`x'` the interpolation is computed the following way:

.. math::
   f(x'_i) = y_j e^{-(x'_i - x_j) b_j},

where :math:`j` is a segment (bin) of :math:`x` to which :math:`x'_i` belongs to, :math:`x_j` is left segment's edge and
:math:`y_j` is the associated value of interpolated function. The :math:`b_j` is thus:

.. math::
   b_j = - \frac{\ln(y_{j+1} / y_j)}{ x_{j+1} - x_j }.

For the points outside of the :math:`[x_0, x_{N-1}]` interval the function value is extrapolated based on first (last)
segment for underflow (overflow).


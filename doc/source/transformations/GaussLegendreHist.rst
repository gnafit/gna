.. _GaussLegendreHist:

GaussLegendreHist: 1d quadrature transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Computes 1d Gauss-Legendre quadrature for a 1d function. Produces histogram as the output.
The transformation is used in pair with :ref:`GaussLegendre <GaussLegendre>`.

Arguments
^^^^^^^^^

1) :ref:`GaussLegendre <GaussLegendre>` instance.

Inputs
^^^^^^

1) ``hist.f`` — function, computed on sample points ``points.x`` of the corresponding :ref:`GaussLegendre <GaussLegendre>` instance.

Outputs
^^^^^^^

1) ``hist.hist`` — the histogram.

.. code:: bash

   ./tests/elementary/test_integral_gh1d.py -s

Implementation
^^^^^^^^^^^^^^

See :ref:`GaussLegendreImplementation` for description.

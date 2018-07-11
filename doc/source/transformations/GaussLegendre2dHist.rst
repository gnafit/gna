.. _GaussLegendre2dHist:

GaussLegendre2dHist: 2d quadrature transformation (1d histogram)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Computes 1d Gauss-Legendre quadrature for a 2d function. Produces 1d histogram as the output.
The transformation is used in pair with :ref:`GaussLegendre2d <GaussLegendre2d>`.

Arguments
^^^^^^^^^

1) :ref:`GaussLegendre2d <GaussLegendre2d>` instance.

Inputs
^^^^^^

1) ``hist.f`` — function, computed on sample points ``points.x`` and ``points.y`` of the corresponding
:ref:`GaussLegendre2d <GaussLegendre2d>` instance.

Outputs
^^^^^^^

1) ``hist.hist`` — the histogram.

Implementation
^^^^^^^^^^^^^^

See  :ref:`GaussLegendre2dImplementation` for description.

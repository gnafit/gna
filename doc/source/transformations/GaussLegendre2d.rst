.. _GaussLegendre2d:

GaussLegendre2d: 2d quadrature transformation (1d histogram)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Computes the sample points and weights needed to integrate 2d function into a 1d histogram. The histogram is binned
along :math:`x` axis while the :math:`y` axis is integrated for a single interval. The integration is done in
combination with :ref:`GaussLegendre2dHist <GaussLegendre2dHist>` transformation.

Arguments
^^^^^^^^^

For a case of :math:`N` bins.

- Case A, using ``std::vector``:

    1) ``edges`` — bin edges for :math:`x` (N+1 doubles).
    2) ``orders`` — integration order for each :math:`x` bin (N of size_t).
    3) ``ymin`` — lower limit for :math:`y` (double).
    4) ``ymax`` — upper limit for :math:`y` (double).
    5) ``yorder`` — integration order for :math:`y` (size_t).

- Case B, using ``double`` pointer:

    1) ``edges`` — bin edges for :math:`x` (N+1 doubles).
    2) ``orders`` — integration order for each :math:`x` bin (N of size_t).
    3) ``bins`` — number of bins (size_t).
    4) ``ymin`` — lower limit for :math:`y` (double).
    5) ``ymax`` — upper limit for :math:`y` (double).
    6) ``yorder`` — integration order for :math:`y` (size_t).

- Case C, using ``std::vector`` (same order):

    1) ``edges`` — bin edges for :math:`x` (N+1 doubles).
    2) ``orders`` — integration order for all bins (size_t).
    3) ``bins`` — number of bins (size_t).
    4) ``ymin`` — lower limit for :math:`y` (double).
    5) ``ymax`` — upper limit for :math:`y` (double).
    6) ``yorder`` — integration order for :math:`y` (size_t).

Outputs
^^^^^^^

1) ``points.x`` — array of sample points :math:`x_i`.
2) ``points.y`` — array of sample points :math:`y_i`.
3) ``points.xedges`` — bin edges.

.. _GaussLegendre2dImplementation:

Implementation
^^^^^^^^^^^^^^

According to Gauss-Legendre approximation the finite integral of order :math:`M (L)` is approximated by:

.. math::
   H = \int\limits_{y_1}^{y_2}dy\int\limits_{a}^{b} f(x,y) dx \approx \sum_{k=1}^{L} \omega_k \sum_{j=1}^{M} \omega_j f(x_j, y_k).

In case of a histogram for each :math:`x` bin :math:`i` with limits :math:`(x_i, x_{i+1})` the integral is given by:

.. math::
   H_i = \int\limits_{y_1}^{y_2}dy\int\limits_{x_i}^{x_{i+1}} f(x, y) dx \approx \sum_{k=1}^{L} \omega_k \sum_{j_i=1}^{M_i} \omega_{ij} f(x_{ij}, y_k),

where :math:`M_i` is integration order for each :math:`x` bin :math:`i` and :math:`L` is integration order for :math:`y`.

For a given set of bin edges and orders the transformation computes sample points (output) and weights (internal). Given
the array of all sample points :math:`x` of size :math:`M` and an array of :math:`y` sample points of size :math:`L` the
function of interest should be computed on each :math:`(x,y)` pair with an output of shape :math:`[M\times L]`. The
resulting matrix should be passed to :ref:`GaussLegendre2dHist <GaussLegendre2dHist>` instance.

For more information on Gauss-Legendre quadrature see https://en.wikipedia.org/wiki/Gaussian_quadrature.



.. _GaussLegendre:

GaussLegendre: 1d quadrature transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Computes the sample points and weights needed to integrate 1d function into a 1d histogram. The integration is done in
combination with :ref:`GaussLegendreHist <GaussLegendreHist>` transformation.

Arguments
^^^^^^^^^

For a case of :math:`N` bins.

- Case A, using ``std::vector``:

    1) ``edges`` — bin edges (N+1 doubles).
    2) ``orders`` — integration order for each bin (N of size_t).

- Case B, using ``std::vector`` (same order):

    1) ``edges`` — bin edges (N+1 doubles).
    2) ``orders`` — integration order for all bins (size_t).
    
- Case C, using ``double`` pointer:

    1) ``edges`` — bin edges (N+1 doubles).
    2) ``orders`` — integration order for each bin (N of size_t).
    3) ``bins`` — number of bins (size_t).

- Case D, using ``std::vector`` (same order):

    1) ``edges`` — bin edges (N+1 doubles).
    2) ``orders`` — integration order for all bins (size_t).
    3) ``bins`` — number of bins (size_t).

Outputs
^^^^^^^

1) ``points.x`` — array of sample points :math:`x_i`.
2) ``points.xedges`` — bin edges.

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/elementary/test_integral_gl1d.py -s

.. _GaussLegendreImplementation:

Implementation
^^^^^^^^^^^^^^

According to Gauss-Legendre approximation the finite integral of order :math:`M` is approximated by:

.. math::
   H = \int\limits_{a}^{b} f(x) dx \approx \sum_{j=1}^{M} \omega_j f(x_j).

In case of a histogram for each bin :math:`i` with limits :math:`(x_i, x_{i+1})` the integral is given by:

.. math::
   H_i = \int\limits_{x_i}^{x_{i+1}} f(x) dx \approx \sum_{j_i=1}^{M_i} \omega_{ij} f(x_{ij}),

where :math:`M_i` is integration order for each bin :math:`i`. 

For a given set of bin edges and orders the transformation computes sample points (output) and weights (internal). The
function of interest should be then computed on :math:`x` and passed to :ref:`GaussLegendreHist <GaussLegendreHist>` instance.

For more information on Gauss-Legendre quadrature see https://en.wikipedia.org/wiki/Gaussian_quadrature.



.. _chi2_transformation:

Chi2
~~~~

Description
^^^^^^^^^^^
Calculates the :math:`\chi^2` value.

Inputs
^^^^^^

  1. Theory :math:`\mu` of size :math:`N`.

  2. Data :math:`x` of size :math:`N`.

  3. Covariance matrix Cholesky_ decomposition :math:`L` of size :math:`N\times N`.

  #. Optionally :math:`\mu_2,x_2,L_2,\dots` of sizes :math:`N_2,\dots`.

Inputs are added via ``add(theory, data, cov)`` method.

.. _Cholesky: https://en.wikipedia.org/wiki/Cholesky_decomposition

Outputs
^^^^^^^

1) ``'chi2'`` — chi-squared value :math:`\chi^2`.

Implementation
^^^^^^^^^^^^^^

For the covariance matrix :math:`V` (symmetric, positively defined), decomposed as

.. math::
  V = L L^T,

where :math:`L` is a lower triangular matrix the tranformation returns the :math:`\chi^2` value:

.. math::
  \chi^2 = (x-\mu)^T V^{-1} (x - \mu).

The exact implementation follows:

.. math::
  y = L^{-1} (x-\mu),

.. math::
  \chi^2 = y^T y.


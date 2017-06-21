Reference guide
==================

Transformations
---------------

Chi2 tranformation
~~~~~~~~~~~~~~~~~~

Inputs
^^^^^^

1) Theory :math:`\mu`

2) Data :math:`x`

3) Covariance matrix Cholesky decomposition :math:`L`

#) Optionally :math:`\mu_2,x_2,L_2,\dots`

Outputs
^^^^^^^

1) Chi-squared value :math:`\chi^2`

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


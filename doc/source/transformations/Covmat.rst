Covmat
~~~~~~

.. attention::

    This is the older version of the covariance matrix calculation. If you care about getting the result, but not
    on the internal details consider using the CovariatedPrediction transformation instead.

Description
^^^^^^^^^^^

Covmat_ class contains three distinct trasformations used to calculate the
covariance matrix.

1) `cov transformation`_ calculates the full covariance matrix based on statistical errors and input Jacobian.
2) `inv transformation`_ calculates the inverse of the input matrix.
3) `cholesky transformation`_ calculates Cholesky decomposition of the covariance matrix, as needed by the Chi2 transformation.

Cov transformation
^^^^^^^^^^^^^^^^^^

Description
"""""""""""

Calculate the final covariance matrix based on:
    * statistical errors
    * extra systematical part based on the variation of the
      prediction :math:`\mu` due to variation of the systematical
      parameters :math:`\eta`.

Inputs
""""""

1) ``'stat'`` — vector with statistical errors :math:`S`.
2) Derivative (Jacobian) :math:`D_1` of the prediction :math:`\mu` over uncertain parameter :math:`\eta_1`.
3) Derivative (Jacobian) :math:`D_i` of the prediction :math:`\mu` over uncertain parameter :math:`\eta_i`.
4) etc.

See ``Derivative`` transformation. Parameters :math:`\eta_i` are meant to be uncorrelated. (To be updated)

Inputs (derivatives) are processed via ``rank1(data)`` method.

Outputs
"""""""

1) ``'cov'`` — full covariance matrix :math:`V`.

Outputs are frozen upon calculation.

Implementation
""""""""""""""

Calculates covariance matrix :math:`V` in the linear approximation:

.. math::
   V = V_\text{stat} + D D^T,

where :math:`V_\text{stat}` is a diagonal matrix with :math:`(V_\text{stat})_{ii} = S_i`.

and :math:`D` is the complete Jacobian:

.. math::
   D = \{ D_1, D_2, \dots \}.

Considering prediction column of size :math:`[N \times 1]` and uncertainties vector of size :math:`M`
the Jacobian :math:`D` dimension is :math:`[N \times M]` and covariance matrix :math:`V` dimension
is :math:`[N \times N]`.

The calculation of :math:`V` is implemented iteratively:

.. math::
   V_i = V_{i-1} + D_i D_i^T, \quad i=1,2,\dots,

where :math:`V_0=V_\text{stat}`.

Inv transformation
^^^^^^^^^^^^^^^^^^

Computes the inverse of a matrix.

Inputs
""""""
1) ``'cov'`` — matrix :math:`V`.

Outputs
"""""""
1) ``'inv'`` — matrix :math:`V^{-1}`.

Cholesky transformation
^^^^^^^^^^^^^^^^^^^^^^^

Computes the Cholesky decomposition of the symmetric positive definite matrix matrix.

Inputs
""""""
1) ``'cov'`` — matrix :math:`V`.

Outputs
"""""""
1) ``'L'`` — lower triangular matrix :math:`L`, such that :math:`V=LL^T`.

**IMPORTANT**: Be sure to use :math:`L` as lower triangular matrix 
(use `numpy.tril` or `triangularView<Eigen::Lower>`). Upper triangular part
may contain unmaintained non-zero elements.

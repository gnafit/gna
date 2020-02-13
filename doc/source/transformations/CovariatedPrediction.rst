.. _CovariatedPrediction:

CovariatedPrediction
~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
CovariatedPrediction class contains three distinct trasformations used to calculate the
compound prediction and the full covariance matrix for it. The trasformations are the following:

1) `Prediction transformation`_ calculates the concatenated prediction based on several input predictions.
2) `Covbase transformation`_ calculates the predefined part of the block covariance matrix based on input covaraince matrices given as is.
3) `Cov transformation`_ calculates the extra systematic part of the covariance matrix based on the known parameters uncertainties.

The result is not the covariance matrix :math:`V` it self, but it's Cholesky decomposition :math:`L` as needed
by the Chi2 transformation.

Prediction transformation
^^^^^^^^^^^^^^^^^^^^^^^^^

Description
"""""""""""

Calculate compound prediction vector :math:`\mu` by concatenating individual predictions :math:`m_i`.

Inputs
""""""
1) Input vector :math:`m_1`.
2) Optional input vector :math:`m_2`.
3) etc.

Vectors :math:`m_i` are added via ``append(obs)`` method.

Outputs
"""""""

1) ``'prediction.prediction'`` — Prediction vector :math:`\mu`.

Implementation
""""""""""""""

Calculates :math:`\mu` as a concatination product of vectors :math:`m_i`:

.. math::
   \mu = \{m_1, m_2, \dots\}.


Covbase transformation
^^^^^^^^^^^^^^^^^^^^^^

Description
"""""""""""

Calculate compound covariance matrix based on statistical uncertainties (diagonal)
, optional covariation matrices for each prediction :math:`m_i`
and between predictions :math:`m_i` and :math:`m_j`. The pull terms are
included here.

This is the constant predefined covariance matrix part: the base.

Inputs
""""""
1) Covariance matrix. Options:

    a) covariance matrix :math:`V_i` for model :math:`m_i`.
    b) cross covariance matrix :math:`V_{ij}` for models :math:`m_i` :math:`m_j`.

2) etc.

Inputs are assigned via ``covariate(cov, obs1, n1, obs2, n2)`` method.

Outputs
"""""""

1) ``'covbase.covbase'`` — basic covariance matrix with optional pull-terms.

Implementation
""""""""""""""

Calculates :math:`V_\text{base}` as a block matrix:

.. math::
   V_\text{base} =
   \begin{pmatrix}
   V_1      & V_{12} & \dots \\
   V_{12}^T & V_{2}  & \dots \\
   \dots    & \dots  & \dots
   \end{pmatrix}.

Returns constructed covariance matrix.
.. Returns the Cholesky decomposition :math:`L_\text{base}`.

Cov transformation
^^^^^^^^^^^^^^^^^^

Description
"""""""""""

Calculate the final covariance matrix based on:
    * predefined covariance base.
    * extra (optional) systematical part based on the variation of the
      prediction :math:`\mu` due to variation of the systematical
      parameters :math:`\eta`.

Inputs
""""""

1) ``'cov.covbase'`` — Base covariance matrix :math:`V_\text{base}`.
2) Optional systematical covariance matrix due to propagation of uncertain parameters.

See ``Jacobian``, ``ParMatrix`` and ``MatrixProduct`` transformations. Parameters :math:`\eta_i` are meant to be uncorrelated.


Outputs
"""""""

1) ``'cov.L'`` — full covariance matrix Cholesky decomposition :math:`L`: :math:`V=LL^T`.

**IMPORTANT**: Be sure to use :math:`L` as lower triangular matrix
(use `numpy.tril` or `triangularView<Eigen::Lower>`). Upper triangular part
may contain unmaintained non-zero elements.

Implementation
""""""""""""""

Calculates covariance matrix :math:`V` in the linear approximation:

.. math::
   V = V_\text{base} + J V_{sys} J^T,

where :math:`J` is the complete Jacobian:

.. math::
   J = \{ J_1, J_2, \dots \}.

Considering prediction column of size :math:`[N \times 1]` and uncertainties vector of size :math:`M`
the Jacobian :math:`J` dimension is :math:`[N \times M]` and covariance matrix :math:`V` dimension
is :math:`[N \times N]`.

Then the Cholesky decomposition is applied to the full covaraince matrix :math:`V`:

.. math::
    V = LL^T

Returns the Cholesky decomposition :math:`L` of :math:`V`.

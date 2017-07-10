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

Vectors :math:`m_i` are added via ``append`` method.

Outputs
"""""""

1) Prediction vector :math:`\mu`.

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
and optional covariation matrices for each predction :math:`m_i`
and between predictions :math:`m_i` and :math:`m_j`.

This is the constant predefined covariance matrix part: the base.

Inputs
""""""
1) Covariance matrix. Options:
   a) covariance matrix :math:`V_i` for model :math:`m_i`.
   b) cross covariance matrix :math:`V_{ij}` for models :math:`m_i` :math:`m_j`.
2) etc.

Inputs are assigned via ``covariate`` method.

Outputs
"""""""

1) Covariance matrix Cholesky decomposition :math:`L_\text{base}`: :math:`V_\text{base}=L_\text{base}L_\text{base}^T`.

**IMPORTANT**: Be sure to use :math:`L_\text{base}` as lower triangular matrix
(use `numpy.tril` or `triangularView<Eigen::Lower>`). Upper triangular part
may contain unmaintained non-zero elements.

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

Returns the Cholesky decomposition :math:`L_\text{base}`.

Cov transformation
^^^^^^^^^^^^^^^^^^

Description
"""""""""""

Calculate the final covariance matrix based on:
    * predefined covariance base.
    * extra systematical part based on the variation of the
      prediction :math:`\mu` due to variation of the systematical
      parameters :math:`\eta`.

Inputs
""""""

1) Base covariance matrix :math:`V_\text{base}`.
2) Derivative (Jacobian) :math:`D_1` of the :math:`\mu` over uncertain parameter :math:`\eta_1`.
3) Derivative (Jacobian) :math:`D_i` of the :math:`\mu` over uncertain parameter :math:`\eta_i`.
4) etc.

See ``Derivative`` transformation. Parameters :math:`\eta_i` are meant to be uncorrelated. (To be updated)

Inputs are processed via ``rank`` method.

Outputs
"""""""

1) Full covariance matrix Cholesky decomposition :math:`L`: :math:`V=LL^T`.

**IMPORTANT**: Be sure to use :math:`L` as lower triangular matrix 
(use `numpy.tril` or `triangularView<Eigen::Lower>`). Upper triangular part
may contain unmaintained non-zero elements.

Implementation
""""""""""""""

Calculates covariance matrix :math:`V` in the linear approximation:

.. math::
   V = V_\text{base} + D D^T,

where :math:`D` is the complete Jacobian:

.. math::
   D = \{ D_1, D_2, \dots \}.

Considering prediction column of size :math:`[N \times 1]` and uncertainties vector of size :math:`M`
the Jacobian :math:`D` dimension is :math:`[N \times M]` and covariance matrix :math:`V` dimension
is :math:`[N \times N]`.

The calculation of :math:`V` is implemented iteratively in terms of rank 1 update:

.. math::
   L_i = \operatorname{rank1}( L_{i-1}, D_i ), \quad i=1,2,\dots,

where :math:`L_0=L_\text{base}`. The function :math:`\operatorname{rank1}` is defined so that for
:math:`V_0 = L_0 L_0^T` the following equation holds:

.. math::
   &V_1 = V_0 + D_1 D_1^T = L_1 L_1^T, \\
   &L_1 = \operatorname{rank1}( L_0, D_1 ).

Returns the Cholesky decomposition :math:`L` of :math:`V`.

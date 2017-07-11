Cholesky
~~~~~~~~

Description
^^^^^^^^^^^
Computes the Cholesky decomposition of the symmetric positive definite matrix matrix.

Inputs
^^^^^^
1) ``'mat'`` — matrix :math:`V`.

Outputs
^^^^^^^
1) ``'L'`` — lower triangular matrix :math:`L`, such that :math:`V=LL^T`.

**IMPORTANT**: Be sure to use :math:`L` as lower triangular matrix
(use `numpy.tril` or `triangularView<Eigen::Lower>`). Upper triangular part
may contain unmaintained non-zero elements.

Cholesky
~~~~~~~~

Description
^^^^^^^^^^^
Computes the Cholesky_ decomposition of the symmetric positive definite  matrix.

.. _Cholesky: https://en.wikipedia.org/wiki/Cholesky_decomposition

Inputs
^^^^^^
1) ``'mat'`` — matrix :math:`V`.

Outputs
^^^^^^^
1) ``'L'`` — lower triangular matrix :math:`L`, such that :math:`V=LL^T`.

**IMPORTANT**: Be sure to use :math:`L` **as lower triangular matrix**
(use numpy.tril_ or `triangularView<Eigen::Lower>`_ ). 
Upper triangular part may contain unmaintained non-zero elements.

.. _triangularView<Eigen::Lower>: https://eigen.tuxfamily.org/dox/classEigen_1_1TriangularView.html
.. _numpy.tril: https://docs.scipy.org/doc/numpy/reference/generated/numpy.tril.html

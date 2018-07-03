.. _Prediction:

Prediction transformation
~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^

The transformation Prediction concatenates the input arrays into single output array.

Inputs
^^^^^^

1) ``prediction.input1`` — array :math:`A_1` of size :math:`N`.
2) ``prediction.input2`` — array :math:`A_2` of size :math:`M`.
3) etc.

New inputs are added via method ``append()``.

Outputs
^^^^^^^

1) ``prediction.prediction`` — array :math:`A`, filled with concatenation result :math:`A=\{A_1, A_2, \dotsc\}` of size
:math:`N+M+\dots`

Implementation
^^^^^^^^^^^^^^

For given input arrays :math:`A_1`, :math:`A_2`, etc the output array is:

.. math::
   A = \{(A_1)_1, \dotsc, (A_1)_N,(A_2)_1, \dotsc, (A_2)_M, \dots\}

Any multidimensional input array is unwrapped into array using Column-major notation (like Fortran):

.. math::
   A_{ij} \rightarrow \{ A_{11}, A_{21}, \dotsc, A_{12}, A_{22}, \dotsc\},

in other words the slices are taken for each column.

Derivative transformation
~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Calculates the derivative of the multidimensional function versus parameter.
Uses finite differences method of 4-th order.

Arguments
^^^^^^^^^

1) ``x`` — parameter instance.
2) ``reldelta`` — finite difference step :math:`h`.

Inputs
^^^^^^

1) ``derivative.y`` — array of size :math:`N`.

Outputs
^^^^^^^

1) ``derivative.dy`` — array of size :math:`N`.

Implementation
^^^^^^^^^^^^^^

The second order finite difference reads as follows:

.. math::
     D_2(h) = \frac{f(x+h) - f(x-h)}{2h}.

The fourth order reads as follows:

.. math::
     \frac{dy}{dx} = D_4(h)
     &= \frac{1}{3} \left(4D_2\left(\frac{h}{2}\right) - D_2(h)\right)
     = \\ &=
     \frac{4}{3h} \left(f\left(x+\frac{h}{2}\right) - f\left(x-\frac{h}{2}\right)\right)
     -  \frac{1}{6h} \left(f\left(x+h\right) - f\left(x-h\right)\right).

for more information see https://en.wikipedia.org/wiki/Finite_difference_coefficient.


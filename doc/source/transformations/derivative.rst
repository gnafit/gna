Derivative transformation
~~~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Calculates the derivative of the multidimensional function versus parameter.
Uses finite differences method of 4-th order.

Arguments
^^^^^^^^^

1) ``'x'`` — parameter instance.
2) ``'reldelta'`` — finite difference step.

Inputs
^^^^^^

1) ``'y'`` — array of size :math:`N`.

Outputs
^^^^^^^

1) ``'dy'`` — array of size :math:`N`.

Implementation
^^^^^^^^^^^^^^

.. math::
     D_2(h) = \frac{f(x+h) - f(x-h)}{2h}

.. math::
     \frac{dy}{dx} = D_4(h)
     = \frac{1}{3} \left(4D\left(\frac{h}{2}\right) - D(h)\right)
     = \\ =
     \frac{1}{3h} \left(f(x+h/2) - f(x-h/2)\right)
     - 4/(3*step) * (h3-h4) - 1/(6*step)*(h1-h2)

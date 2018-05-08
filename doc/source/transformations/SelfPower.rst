.. _SelfPower:

SelfPower
~~~~~~~~~

Description
^^^^^^^^^^^

Computes the result of a coefficient-wise :math:`(x/a)^{\pm x/a}` function.

The objects handles two transformations ``selfpower`` and ``selfpower_inv`` for positive and negative power
respectively. Inputs and outputs are the same.

Inputs
^^^^^^

1. ``selfpower.points`` and ``selfpower_inv.points`` — :math:`x`, input array (not histogram).

Outputs
^^^^^^^

1. ``selfpower.result`` and ``selfpower_inv.result`` — :math:`f`, the function result of the same shape as :math:`x`.

Variables
^^^^^^^^^

1. ``sp_scale`` — :math:`a`, scale to be applied to :math:`x`.

Arguments
^^^^^^^^^

1. ``const char* scalename="sp_scale"`` — the variable name for :math:`a` may be optionally redefined via constructor
   argument.

Tests
^^^^^

Use the following commands for the usage example and testing:

.. code:: bash

   ./tests/detector/test_selfpower.py

Implementation
^^^^^^^^^^^^^^

The result of the ``selfpower`` is:

.. math::
   f_{i} = \left(\frac{x_{i}}{a_{\phantom{i}}}\right)^\frac{x_{i}}{a_{\phantom{i}}}

and for the ``selfpower_inv`` is:

.. math::
   f_{i} = \left(\frac{x_{i}}{a_{\phantom{i}}}\right)^{-\frac{x_{i}}{a_{\phantom{i}}}}.

Since :math:`x` may be multidimensional :math:`i` in these equations may represent hyper-index.

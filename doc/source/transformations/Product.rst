.. _Product:

Product
~~~~~~~

Description
^^^^^^^^^^^
Calculate an elementwise product of several arrays.

Inputs
^^^^^^

1) Array or matrix :math:`a_1`.

2) Array or matrix :math:`a_2`.

N) Array or matrix :math:`a_N`.

The inputs are passed via ``multiply()`` method.

Each input is either a scalar or have to have dimension the same as other non-scalar inputs.

Outputs
^^^^^^^

1) ``'product.product'`` — the product :math:`P`.

Implementation
^^^^^^^^^^^^^^

Computes a product :math:`P`:

.. math::
   P_{ij} = \prod\limits_{k=1}^N (a_k)_{ij}.

If :math:`a_k` is a scalar, then :math:`(a_k)_{ij}=a_k`.


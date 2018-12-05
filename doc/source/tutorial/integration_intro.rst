.. _tutorial_integration_intro:

Introduction
''''''''''''

In order to compute the following integral numerically

.. math::

   H = \int\limits_{a}^{b} f(x)\,dx

one represents it as a sum:

.. math::

   H \approx \sum\limits_{i=1}^K \omega_i f(x_i),

where :math:`K` is the order of integration, :math:`x_i` is a set of points of size :math:`K` on which the function is
computed and :math:`\omega_i` are the weights.

The formula is valid regardless of the method is used. In case the rectangular integration (left) is used with :math:`K=3`,
then :math:`x=\{a, a+\Delta/3, a+2\Delta/3\}` and :math:`\omega_i=\Delta/3`, where :math:`\Delta=b-a`. For :math:`K=3`
and trapezoidal integration :math:`x=\{a, (a+b)/2, b\}` and :math:`\omega=\{\Delta/4, \Delta/2, \Delta/4\}`.

Other methods may be used, such as `Gauss-Legendre
<https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss–Legendre_quadrature>`_ or
`Gauss-Kronrod <https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula>`_ quadratures.
The important point here is that regardless of the method used for the given the order :math:`K` and limits :math:`a`
and :math:`b` the sets of :math:`x_i` and :math:`\omega_i` may be defined once.

The typical problem in calculations is when one needs to compute a set of integrals for each bin to produce a histogram:

.. math::

   H_k = \int\limits_{a_k}^{a_{k+1}} f(x)\,dx.

this integral is approximated in a similar way:

.. math::
   :label: quad1

   H_k \approx \sum\limits_{i_k=1}^{K_k} \omega_{i_k} f(x_{i_k}), \\
   x_{i_k} \in [a_i, a_{i+1}].

Note, that each bin :math:`k` may have different integration precision, different number of integration points and thus
has its own index :math:`i_k`.

The rule may be extended to the higher orders. For example, 2-dimensional integral yielding 2d histogram reads as
follows:

.. math::
   :label: quad2

   H_{km}
   &=
   \int\limits_{a_k}^{a_{k+1}}
   dx
   \int\limits_{b_m}^{b_{m+1}} f(x, y)\,dy
   \approx
    \sum\limits_{i_k=1}^{K_k}
    \sum\limits_{j_m=1}^{M_m}
    \omega_{i_k}^x
    \omega_{j_m}^y
    f(x_{i_k}, y_{j_m}),
    \\
    x_{i_k} &\in [a_i, a_{i+1}],
    \\
    y_{j_m} &\in [b_j, b_{j+1}].

where we have added a set of bins over variable :math:`y` with edges :math:`b_m` and integration orders :math:`M_m`.

Given the formulas :eq:`quad1` and :eq:`quad2` we may now define the integration procedure that involves 2+
transformations:

    #. Sampler transformation. This transformation is provided by the bin edges and required integration orders for each
       bin. The transformation produces the arrays with integration points :math:`x` and weights :math:`\omega`, needed
       to compute an integral for each bin. The values of :math:`x` and :math:`\omega` depend on the quadrature method
       used.
    #. The integrand transformation or calculation chain that implements :math:`y_i=f(x_i)`. It receives array :math:`x`
       as input. The integrand is provided by the user.
    #. Integrator transformation. The transformations receives the output of the integrand transformation :math:`y_i` as
       well as the integration weights from the sampler. Integrator implements the convolution and produces the
       histogram :math:`H` as the output. Integrator implementation does not depend on actual sampler used.


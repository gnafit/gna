NormalToyMC
~~~~~~~~~~~

Description
^^^^^^^^^^^
Generates a random sample distributed according to multivariate normal distribution without correlations.

Uses ``boost::mt19937`` random generator.

The sample becomes frozen upon generation. One has to manually taint the transformation
for the next sample by calling ``nextSample`` method.

Inputs
^^^^^^

1. Average model vector :math:`\mu_1`.
2. First model uncertainties vector :math:`\sigma_1`.
3. Average model vector :math:`\mu_2`.
4. Second model uncertainties vector :math:`\sigma_2`.
5. etc.
6. etc.
   
Inputs are added via ``add(theory, sigma)`` method.

Outputs
^^^^^^^

1. ``'toymc'`` — output vector :math:`x` of size of concatination of :math:`\mu_i`.

Implementation
^^^^^^^^^^^^^^

For the random variable vector :math:`x` of size :math:`N`, distributed around :math:`\mu`
with uncertainties :math:`\sigma` the p.d.f. is:

.. math::
   P(x|\mu) =
   \frac{1}{\sqrt{ (2\pi)^N}\prod\limits_i \sigma_i}
   e^{\displaystyle\frac{1}{2}\sum_i\frac{(x_i-\mu_i)^2}{\sigma^2_i}}.

One can define the vector :math:`z`:

.. math::
   &z_i = \frac{x_i-\mu_i}{\sigma_i},\\
   &x_i = \sigma_i z_i + \mu_i.

Since the transition Jacobian :math:`|dx/dz|=|L|=\prod\limits_i \sigma_i` each :math:`z_i` is distributed
normally with :math:`\sigma=1` with central value of :math:`0`.

The algorithm generates normal vector :math:`z` and transforms it to :math:`x_i=\sigma_i z_i + \mu_i`.

By :math:`\mu` we mean the concatenation of vectors :math:`\mu_i`.

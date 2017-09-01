CovarianceToyMC
~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Generates a random sample distributed according to multivariate normal distribution.

Uses ``boost::mt19937`` random generator.

The sample becomes frozen upon generation. One has to manually taint the transformation
for the next sample by calling ``nextSample`` method.

Warning! Be careful with generatining random numbers with high level of correlation.
Presence of correlation coefficients of order of 0.9 will most likely cause the
distribution width to be invalid. So test the generator prior the actual usage.

Inputs
^^^^^^

1. Average model vector :math:`\mu_1`.
2. First model covariance matrix :math:`V_1` Cholesky decomposition :math:`L_1`.
3. Average model vector :math:`\mu_2`.
4. Second model covariance matrix :math:`V_2` Cholesky decomposition :math:`L_2`.
5. etc.
6. etc.
   
Inputs are added via ``add(theory, cov)`` method.

Outputs
^^^^^^^

1. ``'toymc'`` — output vector :math:`x` of size of concatination of :math:`\mu_i`.

Implementation
^^^^^^^^^^^^^^

For the random variable vector :math:`x` of size :math:`N`, distributed around :math:`\mu`
with covariance matrix :math:`V` the p.d.f. is:

.. math::
   P(x|\mu) =
   \frac{1}{\sqrt{ (2\pi)^N |V| }}
   e^{\frac{1}{2} \displaystyle(x-\mu)^T V^{-1} (x-\mu) }.

For the decomposed covariance matrix :math:`V=LL^T` the exponent power reads:

.. math::
   \left((x-\mu) L^{-1}\right)^T L^{-1} (x-\mu).

One can define the variable :math:`z`:

.. math::
   &z = L^{-1} (x-\mu),\\
   &x = Lz + \mu.

Since the transition Jacobian :math:`|dx/dz|=|L|=\sqrt{|V|}` each :math:`z_i` is distributed
normally with :math:`\sigma=1` with central value of :math:`0`.

The algorithm generates normal vector :math:`z` and transforms it to :math:`x=Lz + \mu`.

By :math:`\mu` we mean the concatenation of vectors :math:`\mu_i`.

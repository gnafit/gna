Poisson
~~~~~~~

Description
^^^^^^^^^^^
Calculates the Poisson loglikelihood function value

Inputs
^^^^^^

1) Theory :math:`\mu` of size :math:`N`

2) Data :math:`x` of size :math:`N`

#) Optionally :math:`\mu_2,x_2,\dots` of sizes :math:`N_2,\dots`

Outputs
^^^^^^^

1) -2 Poisson loglikelihood function value

Implementation
^^^^^^^^^^^^^^

Formula for Poisson likelihood:

.. math::
  L(x|\mu) = \prod_{i=1}^{N} \frac {(\mu_i^{x_i}  e^{-\mu_i})}{x_i!} 

In source code there is used the natural logarithm of :math:`P(x|\mu)`, so formula is

.. math::
  \log L(x|\mu) = \sum_{i=1}^{N} {(x_i \log(\mu_i)  - \mu_i -  log(x_i!))}

where :math:`N` is a size vectors :math:`\mu` and :math:`x` 

In the third part of this sum there is :math:`x!` - the function that increase so fast and can cause overflow. 

There are two ways to deal with it. By default the program uses natural logarithm of Gamma function:

.. math::
  \Gamma(x) = (x - 1)!

It also can be used the following approximation

.. math:: 
  \log(x!) \approx x log(x)

You have to add ``--ln-approx`` parameter in python script to apply it.

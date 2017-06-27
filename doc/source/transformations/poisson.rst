Poisson
~~~~~~~

Description
^^^^^^^^^^^
Calculates the Poisson loglikelihood function value

Inputs
^^^^^^

1) Theory :math:`\mu` of size :math:`N`

2) Data :math:`x` of size :math:`N`

3) Covariance matrix (that is actually not used but it is necessary for universal interface)

#) Optionally :math:`\mu_2,x_2,L_2,\dots` of sizes :math:`N_2,\dots`

Outputs
^^^^^^^

1) -2 Poisson loglikelihood function value

Implementation
^^^^^^^^^^^^^^

Formula for Poisson likelihood:

.. math::
  L(x|\mu) = \frac {(\mu^{x}  e^{-\mu})}{x!} 

In source code there is used the natural logarithm of :math:`P(x|\mu)`, so formula is

.. math::
  \log L(x|\mu) = x \log(\mu)  - \mu -  log(x!)

In the third part of this sum there is :math:`x!` - the function that increase so fast and can cause overflow. 

There are two ways to deal with it. By default the program uses natural logarithm of Gamma function:

.. math::
  \Gamma(x) = (x - 1)!

It also can be used the following approximation

.. math:: 
  \log(x!) = x log(x)

You have to add :math:`ln\_approx` parameter in python script to apply it.

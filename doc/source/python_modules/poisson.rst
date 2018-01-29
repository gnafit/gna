Poisson
^^^^^^^

*Poisson* is a module that provides access to the :ref:`poisson_transformation` 
transformation to set Poissonian log likelihood as a test statistic.

It accepts the following arguments:

    * ``name`` -- name that would be assosiated with the test statistic.
    * ``analysis`` -- the analysis from which a theoretical predictions and data would be read.
    * ``--ln-approx`` -- whether to use approximate gamma function logarithm
      with Stirling formulae or not.

Usage (for more realistic look in ``examples/`` or ``scripts/``):

.. code-block:: bash

   python gna ...  -- poisson my_poisson my_analysis --ln-approx

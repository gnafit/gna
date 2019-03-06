Chi2
^^^^

*Chi2* is a module that provides access to the :ref:`chi2_transformation` 
transformation to set a :math:`\chi^2` test statistic.

It accepts the following **positional arguments**:

    * ``name`` -- name that would be assosiated with the test statistic.
    * ``analysis`` -- the analysis from which a theoretical predictions, data
      and covariance matrices would be read.

Usage (for more realistic look in ``examples/`` or ``scripts/``):

.. code-block:: bash

   python gna ...  -- chi2 my_chi2 my_analysis

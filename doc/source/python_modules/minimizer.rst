Minimizer
^^^^^^^^^

The `minimizer` module is used to set up minimizer fot a given test statistic (quite surprising, right).
The following arguments are provided, **positional** first:

    * ``name`` -- the name that would be assosiated with the minimizer object.
    * ``type`` -- the type of minimization procedure to be used. The only
      choice for now is MINUIT bundled with ROOT
    * ``statistic`` -- the test statistic to be minimized. Should be created
      **prior** to the minimezer.
    * ``par`` -- the sequence of parameters to minimize over it. Takes
      arbitrary number of parameters alongside with namespaces. All parameters
      from each namespace that influence test statistic are to be added to
      minimezer.
    * ``-s, --spec`` -- special options for a minimization parameters, should
      be valid YAML. The following options are available:

      + ``value`` -- set central value for parameter.
      + ``limits`` -- constrain parameter minimization to given limits.
      + ``step`` -- initial step for minimization of given parameter.
      + ``fixed`` -- fixes parameter value, that parameter is excluded from
        minimization.

Usage (for real world example look into ``examples/`` folder):

.. code-block:: bash

  python gna ... -- minimizer my_name my_type my_statistic par1 par2 namespace1 -s '{par1: {limits: [0, 100]}, par2: {fixed: True}}' 
   

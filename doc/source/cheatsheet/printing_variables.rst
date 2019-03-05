Printing variables
^^^^^^^^^^^^^^^^^^

Recursive variable list may be printed to the terminal by the ``gna.parameters.printer`` module. The
``print_parameters(ns)`` function accepts the namespace to print:

.. code-block:: python

    from gna.parameters.printer import print_parameters
    print_parameters( env.globalns )

The example output from the ``tests/bundle/detector_dbchain.py`` (line 98)::

    Variables in namespace ''
      weight_nominal                =         1 │ [fixed]
      weight_pull0                  =         0 │          0±         1
      weight_pull1                  =         0 │          0±         1
      weight_pull2                  =         0 │          0±         1
      weight_pull3                  =         0 │          0±         1
      Eres_a                        =  0.014764 │   0.014764± 0.0044292 [        30%]
      Eres_b                        =    0.0869 │     0.0869±   0.02607 [        30%]
      Eres_c                        =    0.0271 │     0.0271±   0.00813 [        30%]
    Variables in namespace 'AD11'
      OffdiagScale                  =         1 │          1±      0.04 [         4%]
      escale                        =         1 │          1±     0.002 [       0.2%]
    Variables in namespace 'AD21'
      OffdiagScale                  =         1 │          1±      0.04 [         4%]
      escale                        =         1 │          1±     0.002 [       0.2%]
    Variables in namespace 'AD31'
      OffdiagScale                  =         1 │          1±      0.04 [         4%]
      escale                        =         1 │          1±     0.002 [       0.2%]

This example prints the variables of the :ref:`bundlechain_v01` bundle:
    1. Global namespace with
        a) 5 energy scale parameters: 1 fixed and 4 correction weights,
        b) 3 energy resolution parameters.
    2. Three nested namespaces for the detectors ``AD11``, ``AD21`` and ``AD31`` with
        a) IAV off-diagonal scale parameter,
        b) Linear scale correction.

The following columns are printed:
    1. Variable name.
    2. Current value.
    3. Central value or ``[fixed]``.
    4. Standard deviation.
    5. Relative deviation if central value is nonzero.



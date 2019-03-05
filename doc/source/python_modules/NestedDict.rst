.. _NestedDict:

NestedDict
^^^^^^^^^^

Overview
""""""""

:ref:`NestedDict` is a helper class very similar to the python ``dict`` or ``OrderedDict`` with the following extra features:

1. :ref:`NestedDict` enables attribute access as well as key access to the stored items:

   .. code-block:: python

       nd = NestedDict( a=1, b=2 )
       nd['a']
       # 1
       nd.a
       # 1

2. :ref:`NestedDict` enables nesting (unlimited):

   * Each stored dictionary will be converted to the :ref:`NestedDict` instance.
   * Assigning to ``nd['a.b']`` will create :ref:`NestedDict` ``nd['a']`` and assign ``nd['a']['b']``.
   * Assigning to ``nd.a.b`` also works, but only for existing ``nd.a``.
   * Nested :ref:`NestedDict` may be also  created by calling ``nd('a')`` or ``nd('a.b')`` or, etc.
   * Parent dictionary may be accessed by calling ``nd.parent()`` method.

3. Usual ``dict`` methods like ``set``, ``setdefault``, ``get``, ``keys``, ``values`` and ``items`` are also supported.

It should be noted that the keys ``set``, ``setdefault``, ``get``, ``keys``, ``values`` and ``items`` may be accessed
only via ``[]`` operator or the methods themselves, but not via attribute access.

Example
"""""""

An example of the configuration of the  :ref:`bundlechain_v01` via :ref:`NestedDict` may be found below:

.. code-block:: python

    cfg = NestedDict()
    cfg.detector = NestedDict(
            bundle = 'bundlechain_v01',
            detectors = [ 'AD11', 'AD21', 'AD31' ],
            chain = [ 'iav', 'nonlinearity', 'eres', 'rebin' ]
            )
    cfg.detector.nonlinearity = NestedDict(
            bundle = 'detector_nonlinearity_db_root_v01',
            names = [ 'nominal', 'pull0', 'pull1', 'pull2', 'pull3' ],
            filename = 'output/detector_nl_consModel_450itr.root',
            uncertainty = 0.2*percent,
            uncertainty_type = 'relative'
            )
    cfg.detector.iav = NestedDict(
            bundle = 'detector_iav_db_root_v01',
            parname = 'OffdiagScale',
            uncertainty = 4*percent,
            uncertainty_type = 'relative',
            ndiag = 1,
            filename = 'data/dayabay/tmp/detector_iavMatrix_P14A_LS.root',
            matrixname = 'iav_matrix'
            )
    cfg.detector.eres = NestedDict(
            bundle = 'detector_eres_common3',
            # pars: sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 ),
            values  = [ 0.014764, 0.0869, 0.0271 ],
            uncertainties = [30.0*percent]*3,
            uncertainty_type = 'relative'
            )
    cfg.detector.rebin = NestedDict(
            bundle = 'rebin',
            rounding = 3,
            edges = [ 0.0, 5.0, 10.0 ]
            )

Loading configuration from file
"""""""""""""""""""""""""""""""

The ``configurator`` module contains the ``configurator`` function that can read any python file and return it as
:ref:`NestedDict`. The following variables may be used within the :ref:`NestedDict` file:

  + ``percent`` equals to ``0.01``.
  + ``numpy`` to access numpy functions.
  + ``load`` to load nested configuration files. ``load`` is a shortcut for ``configurator``.

The file may be located anywhere. Also, all the created ``dict`` instances will be converted to :ref:`NestedDict`.

Example code is below:

.. code-block:: python

    from gna.configurator import configurator
    cfg = configurator( 'path/to/python/file.py' )

See also
""""""""

There testing files that may be used as example:

.. code:: bash

   # reading configuration files:
   ./tests/elementary/test_cfgloader.py
   # assigning items:
   ./tests/elementary/test_cfg.py



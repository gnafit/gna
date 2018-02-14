.. _bundlesum_v01:

Bundle sum (version 1)
^^^^^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

The bundle builds a sum of outputs of other bundles.

Scheme
""""""

1. For each bundle call ``execute_bundle()`` function passing common namespace, individual namespaces.
   The namespaces are populated by the parameters and transformations created by the nested bundles automatically.
2. For each namespace sum up the outputs of all the bundles by namespace name.

The :ref:`Sum` transformation is used for the summation.

Outputs
"""""""

The transformation provides the :ref:`Sum` and it's output for each namespace.

.. code-block:: python

   self.transformations_out[name] = osum.sum
   self.outputs[name]             = osum.sum.outputs['sum']

Observables
"""""""""""

If ``observable`` field is specified the bundle adds the output of each sum to the corresponding namespace as
observable.

Required fields:

  - ``bundle`` for bundle name.
  - ``list`` for the list of the bundles to execute. For each name in the list there should exist a nested NestedDict of
    the same name. This dictionary will be passed as configuration for the ``execute_bundle``.

Optional fields:
  - ``observable`` -- observable name to use it to add the sum output to each namespace.

Configuration
"""""""""""""

.. code-block:: python

    bkg = NestedDict()
    # the bundle name
    bkg.bundle = 'bundlesum_v01'
    # the list of bundles to load
    bkg.list = [ 'bkg1', 'bkg2' ]
    # the observable name to create (optional)
    bkg.observable = 'bkg_total'

    # first background from the list
    bkg.bkg1 = NestedDict(
        # first background specification
        # bundle=
    )

    # second background from the list
    bkg.bkg2 = NestedDict(
        # second background specification
        # bundle=
    )

Testing scripts
"""""""""""""""

The bundle is tested within :ref:`bkg_weighted_hist_v01` bundle test script:
.. code-block:: sh

    ./tests/bundle/bkg_weighted_hist_v01.py -s




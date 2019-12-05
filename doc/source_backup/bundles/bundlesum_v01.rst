.. _bundlesum_v01:

bundlesum_v01 -- sum bundles outputs (version 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

The bundle builds a sum of outputs of other bundles.

Scheme
""""""

1. For each bundle call ``execute_bundle()`` function passing common namespace, individual namespaces.
   The namespaces are populated by the parameters and transformations created by the nested bundles automatically.
2. For each namespace sum up the outputs of all the bundles by namespace name.

The :ref:`Sum` transformation is used for the summation.

Arguments
"""""""""

The bundle supports an argument ``listkey`` which by default is ``bundlesum_list``. It is an address to retrieve the
list of bundles to load and sum. Therefore the bundle may be initialized with extended name:

.. code-block:: python

    cfg = NestedDict(
       bundle = 'bundlesum_v01:mylistname',
       mylistname = [ 'bundle1', 'bundle2' ],
       # other options
    )

In this key the bundle will use ``mylistname`` as the bundle list to read.

Outputs
"""""""

The transformation provides the :ref:`Sum` and it's output for each namespace.

.. code-block:: python

   self.transformations_out[name] = osum.sum
   self.outputs[name]             = osum.sum.outputs['sum']

If ``chaininput`` configuration option is provided, an open input is provided for each namespace.

.. code-block:: python

   inp = osum.add(chaininput)
   """Save unconnected input"""
   self.inputs[name]             = inp
   self.transformations_in[name] = osum

Observables
"""""""""""

If ``observable`` field is specified the bundle adds the output of each sum to the corresponding namespace as
observable.

Required fields:

  - ``bundle`` for bundle name.
  - ``list`` for the list of the bundles to execute. For each name in the list there should exist a nested NestedDict of
    the same name. This dictionary will be passed as configuration for the ``execute_bundle``.

Optional fields:
  - ``observable`` (string) -- observable name to use it to add the sum output to each namespace.
  - ``chaininput`` (string) -- open input name. If provided, the bundle will initialize additional sum input and
    register it in `inputs`.
  - ``debug`` (bool). If true the bundle will print the debug output: what is being summed up.

Configuration
"""""""""""""

.. code-block:: python

    bkg = NestedDict()
    # the bundle name
    bkg.bundle = 'bundlesum_v01'
    # the list of bundles to load
    bkg.bundlesum_list = [ 'bkg1', 'bkg2' ]
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




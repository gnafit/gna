.. _rebin_bundle:

Rebin
^^^^^

Overview
""""""""

The bundle rebins the input histogram to new bin edges. Each new bin edge should coincide to a given precision with one of the old
bin edges.

The bundle is a wrapper and configurator for the :ref:`Rebin` transformation which should be referred for further
documentation.

Scheme
""""""

1. For each of the provided namespaces create :ref:`rebin_bundle` instance according to the configuration.

Configuration
"""""""""""""

.. code-block:: python

   cfg = NestedDict(
           # bundle name
           bundle = 'rebin',
           # the precision of matching bin edges (decimal places after period)
           rounding = 3,
           # new bin edges
           edges = [ 0.0, 5.0, 10.0 ]
           )

Testing scripts
"""""""""""""""

There is now individual testing script for :ref:`rebin_bundle`. Nevertheless it is included in the
:ref:`bundlechain_v01` testing script:

.. code-block:: sh

    tests/bundle/detector_dbchain.py



.. _rebin_bundle:

rebin -- rebin histograms
^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

The bundle rebins the input histogram to new bin edges. Each new bin edge should coincide to a given precision with one of the old
bin edges.

The bundle is a wrapper and configurator for the :ref:`Rebin` transformation which should be referred for further
documentation.

Scheme
""""""

1. For each of the provided namespaces create :ref:`rebin_bundle` instance according to the configuration.

Inputs, outputs and observables
"""""""""""""""""""""""""""""""

The bundle provides the input and output of the :ref:`Rebin` by namespace name. The observable ``'rebin'`` is also
defined for the corresponding namespace:

.. code-block:: python

    self.inputs[ns.name]              = rebin.rebin.histin
    self.outputs[ns.name]             = rebin.rebin.histout
    ns.addobservable('rebin', rebin.rebin.histout, ignorecheck=True)

.. attention::

    When observable is added no check is perfomed whether the input is connected. The DataType and Data are not
    initialized.

Configuration
"""""""""""""

Optional options:
  - ``observable`` (bool or string). If provided, the observable is added for each output to th relevant namespace. If
    true the name 'rebin' is used.

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

There is no individual testing script for :ref:`rebin_bundle`. Nevertheless it is included in the
:ref:`bundlechain_v01` testing script:

.. code-block:: sh

    tests/bundle/detector_dbchain.py



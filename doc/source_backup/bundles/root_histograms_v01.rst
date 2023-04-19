.. _root_histograms_v01:

root_histograms_v01 -- read histograms from a ROOT file (version 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

The bundle ``root_histograms_v01`` enables user to load histograms from a ROOT file. The bundle was created in order to
provide Daya Bay backgrounds data.

Scheme
""""""

The bundle ignores provided namespaces. It operates its own list of namespaces based on configuration.

1. Read a set of histograms from a ROOT file.
2. For each histogram are dedicated namespace is created in the ``common_namespace``.
3. Each histogram is converted to a ``Histogram`` object with open output. The corresponding transformation and output
   are stored by the same name as the namespace.

Outputs
"""""""

The output of the :ref:`Histogram` is provided for each variant.

.. code-block:: python

    self.transformations_out[var] = hist.hist
    self.outputs[var]             = hist.hist.hist

Configuration
"""""""""""""

Below one may find three example configurations for ``root_histograms_v01``:
  - ``spectra1`` reads the histogram ``'hist'`` from the file ``'output/sample_hists.root'``.
  - ``spectra2`` reads the histograms ``'hist_G1'``, ``'hist_G2'`` and ``'hist_G3'`` from the file
    ``'output/sample_hists.root'``.
  - ``spectra3`` reads the histograms ``'hist_G1_D1'``, ``'hist_G1_D2'``, ``'hist_G2_D3'`` and ``'hist_G3_D4'`` from the
    file ``'output/sample_hists.root'``.

The names are resolved by the following scheme:
  - If only ``format`` field is provided, it is used as a name of the histogram.
  - In case ``variants`` are provided:

    * if ``variants`` is a list of strings for each variant from ``variants`` a namespace is created with
      ``name=variant``, a histogram is read from a file with name formatted from ``format`` in which first occurrence of
      ``{self}`` is replaced by variant.
    * if ``variants`` is a dictionary the procedure is the same with the only difference: dictionary key is used as
      namespace names and dictionary values are used to substitute ``{self}`` in the format string.

    See https://pyformat.info for more information on formatting.

The following keys are recognized:
  - ``filename`` -- the filename to read.
  - ``format`` -- the histogram name format.

Optional options:
  - ``variants`` (list or dict) -- list of values or dictionary wit key-value pairs. Each value is applied to the format field.
  - ``normalize`` (bool) -- if True the histogram will be normalized so the integral=1.
  - ``observable`` (string). If provided, the observable is added for each output to th relevant namespace.

.. code-block:: python

    detectors = [ 'D1', 'D2', 'D3', 'D4' ]
    groups=NestedDict([
            ('G1', ['D1', 'D2']),
            ('G2', ['D3']),
            ('G3', ['D4'])
            ])

    spectra1 = NestedDict(
            # bundle name
            bundle = 'root_histograms_v01',
            # filename to read
            filename =  'output/sample_hists.root',
            # histogram name to read
            format = 'hist',
            )
    spectra2 = NestedDict(
            # bundle name
            bundle = 'root_histograms_v01',
            # filename to read
            filename =  'output/sample_hists.root',
            # formattable histogram name to read
            format = 'hist_{self}',
            # variants to pass to the format
            variants = groups.keys()
            )
    spectra3 = NestedDict(
            # bundle name
            bundle = 'root_histograms_v01',
            # filename to read
            filename =  'output/sample_hists.root',
            # formattable histogram name to read
            format = 'hist_{self}',
            # variants to pass to the format (dictionary)
            variants = dict([
                ( 'D1', 'G1_D1' ),
                ( 'D2', 'G1_D2' ),
                ( 'D3', 'G2_D3' ),
                ( 'D4', 'G3_D4' ),
                ])
            )

Testing scripts
"""""""""""""""

.. code-block:: sh

    ./tests/bundle/root_histograms_v01.py



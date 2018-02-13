.. _hist_mixture_v01_bundle:

Weighted histogram mixture (version 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

The bundle ``hist_mixture_v01`` enables user to mix :math:`N` histograms with :math:`N-1` weights.

Scheme
""""""

1. Read a set of input spectra. The spectra have names.
2. In each namespace:
   - For each of :math:`N-1` spectra create a variable ``frac_<name>`` and initialize it with value and uncertainty from
    configuration file. The last weight is defined as :math:`1-w_1-\dots-w_{N-1}`.
   - Create a weighted sum of :math:`N` histograms with corresponding weights.

Configuration
"""""""""""""

The configuration contains two main items:
1. ``fractions`` -- dictionary with key-value pairs, where values are :math:`N-1` default weight values and their
   uncertainties.
2. ``spectra`` -- dictionary with :math:`N` key-value pairs, where value is a bundle with output histogram. In the
   example below ``root_histograms_v01`` is used. Keys in spectra should correspond to keys in fractions.

See ref  :ref:`root_histograms_v01_bundle` for the explanation of the relevant bundle configuration.

.. code-block:: python

    spectra = NestedDict(
        # bundle name
        bundle = 'hist_mixture_v01',
        # fractions for N-1 spectra with uncertainties
        fractions = uncertaindict(
            li = ( 0.90, 0.05, 'relative' )
            ),
        # N spectra to mix
        spectra = NestedDict([
            ('li', NestedDict(
                bundle = 'root_histograms_v01',
                filename   = cfg.filename,
                format = 'hist_G1_D1',
                normalize = True,
                )),
            ('he', NestedDict(
                bundle = 'root_histograms_v01',
                filename   = cfg.filename,
                format = 'hist_G2_D3',
                normalize = True,
                )),
            ])
        )

Testing scripts
"""""""""""""""

The bundle ``hist_mixture_v01`` is tested within ``bkg_weighted_hist_v01`` testing script:

.. code-block:: sh

    ./tests/bundle/bkg_weighted_hist_v01.py



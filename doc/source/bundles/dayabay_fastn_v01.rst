.. _dayabay_fastn_v01_bundle:

Fast neutron background for Daya Bay  (version 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

The bundle ``dayabay_fastn_v01`` implements fast neutron background spectrum as parametrized for Daya Bay analysis:

.. math::
   \frac{dN(E)}{dE} = \frac{\left(\frac{E}{a}\right)^{-\frac{E}{a}}}{\int\limits_{E_\text{min}}^{E_\text{max}} \left(\frac{y}{a}\right)^{-\frac{y}{a}}}dy.

The output histogram is then the integral of the above function over the bin:

.. math::
   N_i = \int\limits_{E_i}^{E_{i+1}}\frac{dN(E)}{dE}dE.


The spectrum then may be used within  :ref:`bkg_weighted_hist_v01` bundle.

Scheme
""""""

For each parameter in ``pars``:

  - initialize the variable according to the ``format``.
  - create the namespace of the same name and provide the output histogram.

The bundle uses :ref:`SelfPower` transformation for the function, GaussLegendre integrator for the integration and
:ref:`Normalize` transformation for the normalization over the range.

Parameters
""""""""""

1. A shape parameter is defined according to ``formula`` for each field of the ``pars``. See Configuration_.

Outputs
"""""""

For each parameter name the bundle provides the :ref:`Normalize` transformation as output transformation and
``normalize.out`` as the output histogram.

.. code-block:: python

    self.transformations_out[ns.name]    = normalize.normalize
    self.outputs[ns.name]                = normalize.normalize.out

Configuration
"""""""""""""

Required parameters:

  - ``bundle`` -- bundle name.
  - ``formula`` -- the format for the variable names and locations. Formula may contain the group types that will be
    replaced by the correct strings from the group specification. For the example below the following categories are
    defined: ``{exp}``, ``{site}`` or ``{det}``. For example, for detector ``'D1'`` the site is ``'G1'``, exp is
    ``'testexp'`` and det is ``'D1'``. For the group ``'G1'`` the exp will be determined as well.
  - ``groups`` -- categories specification.
  - ``bins`` -- the output histogram binning definition.
  - ``order`` -- the integration order for each bin separately (array) or for all the bins in the same time (int).
  - ``pars`` -- dictionary with values and uncertainties for the parameter :math:`a`.

In the example below the formula will initialize a namespace ``'fastn_shape'`` and store the parameters named after each
site: ``'G1'``, ``'G2'`` and ``'G3'``. The output histogram will be created for each site respectively.

In case the formula is ``{site}.fastn_shape`` the parameters will be located in a separate namespaces for each site. The
parameter name will be ``'fastn_shape'``.

.. code-block:: python

    # Detectors specifications
    detectors = [ 'D1', 'D2', 'D3', 'D4' ]
    # Dictionary with detector categories and groups specification
    groups=NestedDict(
        exp  = { 'testexp': detectors },
        det  = { d: (d,) for d in detectors },
        site = NestedDict([
            ('G1', ['D1', 'D2']),
            ('G2', ['D3']),
            ('G3', ['D4'])
            ])
        )
    # the spectrum configuration
    spectra = NestedDict(
        # bundle name
        bundle='dayabay_fastn_v01',
        # the parameter naming format
        formula='fastn_shape.{site}',
        # groups and categories specification (optional)
        groups=groups,
        # binning
        bins = N.linspace(0.0, 12.0, 241),
        # the range for the spectrum normalization [Emin, Emax)
        normalize=(0.7, 12.0),
        #
        # integration order (Gauss-Legendre):
        #   - common order for all the bins.
        #   - array of orders for each individual bin.
        order=2,
        #
        # The value of the shape parameter for each detector or detector group.
        #
        pars=uncertaindict(
           [ ('G1', (70.00, 0.1)),
             ('G2', (60.00, 0.05)),
             ('G3', (50.00, 0.2)) ],
            mode='relative',
            ),
        )

Testing scripts
"""""""""""""""

The bundle ``dayabay_fastn_v01`` is tested within ``bkg_weighted_hist_v01`` testing script:

.. code-block:: sh

    ./tests/bundle/bkg_weighted_hist_v01.py



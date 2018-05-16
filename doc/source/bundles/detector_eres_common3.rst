.. _detector_eres_common3:

Energy resolution (version 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

This class is a simple wrapper and configurator for the  :ref:`EnergyResolution` transformation. The energy resolution
parameters' uncertainties :math:`a`, :math:`b` and :math:`c` are considered to be uncorrelated between parameters and
fully correlated between detectors.

Scheme
""""""

1. Create  :ref:`EnergyResolution` instance. It will have as many inputs as the number of provided namespaces.
2. Create :math:`a`, :math:`b` and :math:`c` parameters in a common namespace.

.. figure:: ../../img/eres_scheme.png
   :scale: 25 %
   :align: center

   Energy resolution bundle scheme.

Parameters
""""""""""

:ref:`EnergyResolution` parameters :math:`a`, :math:`b` and :math:`c`.


Configuration
"""""""""""""

.. code-block:: python

    cfg = NestedDict(
            # bundle name
            bundle = 'detector_eres_common3',
            # parameters a, b and c respectively for sigma_e/e = sqrt( a^2 + b^2/E + c^2/E^2 )
            values  = [ 0.014764, 0.0869, 0.0271 ],
            # uncertainty on each parameter
            uncertainties = [30.0*percent]*3,
            # uncertainty type (absolute/relative)
            uncertainty_type = 'relative'
            )

Testing scripts
"""""""""""""""

There is now individual testing script for  :ref:`detector_eres_common3`. Nevertheless it is included in the
:ref:`bundlechain_v01` testing script:

.. code-block:: sh

    tests/bundle/detector_dbchain.py -s



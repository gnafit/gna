.. _concatenate_bundle:

concatenate -- concatenate arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Overview
""""""""

The bundle concatenates the outputs of all the namespaces into a common array.

The bundle is a wrapper and configurator for the :ref:`Prediction` transformation which should be referred for further
documentation.

Inputs, outputs and observables
"""""""""""""""""""""""""""""""

The bundle provides the output of the :ref:`Prediction` by the ``common_namespace`` name. 
The observable is optionally added according to the configuration.

.. code-block:: python

    self.transformations_out[self.common_namespace.name] = concat.prediction
    self.outputs[self.common_namespace.name]             = concat.prediction.prediction

.. attention::

    When observable is added no check is perfomed whether the input is connected. The DataType and Data are not
    initialized.

Configuration
"""""""""""""

The bundle has the only optional option ``observable``. Concatenate will add an observable with a provided name in the
``common_namespace``.

.. code-block:: python

    cfg = NestedDict(
        bundle     = 'concatenate',
        observable = 'concat'
        )

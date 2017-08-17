Identity transformation
~~~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
Copies input to output as is.

Inputs
^^^^^^

1) ``'source'`` — array of size :math:`N`.

Outputs
^^^^^^^

1) ``'target'`` — array of size :math:`N`.

Implementation
^^^^^^^^^^^^^^
The class is needed in order to overcome the GNA limitation:
the single not connected anywhere transformation is not properly initialized.
Therefore Identity transformation may be used to emulate connection.

.. _Points:

Points transformation
~~~~~~~~~~~~~~~~~~~~~

Description
^^^^^^^^^^^
'Static' transformation. Represents an array of specified the shape.

Arguments
^^^^^^^^^

* ``std::vector`` or
* ``double*`` array and size or
* ``double*`` and ``std::vector`` with shape definition.

In Python ``Points`` instance may be constructed from the numpy array:

.. code-block:: ipython

   from gna.constructors import Points
   p = Points(array)

Outputs
^^^^^^^

1) ``points.points`` — static array of kind Points.


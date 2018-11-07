.. _Histogram:

Histogram
~~~~~~~~~

Description
^^^^^^^^^^^
'Static' transformation. Represents the histogram.

See also ``HistEdges`` and ``Rebin`` transformations.

Arguments
^^^^^^^^^

* ``size_t`` — number of bins :math:`n`
* ``double*`` — array with bin edges of size :math:`n+1`
* ``double*`` — array with bin heights of size :math:`n`

In Python ``Histogram`` instance may be constructed from two numpy arrays:

.. code-block:: ipython

   from constructors import Histogram
   h = Histogram(edges, data)

Outputs
^^^^^^^

1) ``hist.hist`` — static array of kind Histogram.


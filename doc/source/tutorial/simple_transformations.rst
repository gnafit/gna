Simple transformations
^^^^^^^^^^^^^^^^^^^^^^

Here we will review several simple transformations. Simple means that they do not depend on any variables and have no
inputs. There are currently three transformations `Points`, `Histogram` and `Histogram2d` that enable user to initialize
the input data (arrays or histograms) in forms of transformation outputs.

Points
""""""

The :ref:`Points <Points>` transformation is used to represent 1d/2d array as transformation output. The ``Points``
instance is created with ``numpy`` array passed as input:

.. literalinclude:: ../../../macro/tutorial/basic/01_points.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 7,10,12,14
    :caption: :download:`01_points.py <../../../macro/tutorial/basic/01_points.py>`

The code produces the following output:

.. code-block:: text
    :linenos:

    Output: [out] points: array 2d, shape 3x4, size  12
    DataType: array 2d, shape 3x4, size  12
    Data:
     [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]]


Histogram
"""""""""

The :ref:`Histogram` transformation stores a 1-dimensional histogrammed data. It is very similar to the 1d version of
`Points` with the only difference: its DataType stores the bin edges.

.. literalinclude:: ../../../macro/tutorial/basic/02_hist.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 12,17,20,21,23
    :caption: :download:`02_hist.py <../../../macro/tutorial/basic/02_hist.py>`

On line 21 `datatype.edges` C++ vector is converted to to the python list.

The code produces the following output:

.. code-block:: text
    :linenos:

    Output: [out] hist: hist,  12 bins, edges 1.0->7.0, width 0.5
    DataType: hist,  12 bins, edges 1.0->7.0, width 0.5
    Bin edges: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    Data: [  0. 100. 324. 576. 784. 900. 900. 784. 576. 324. 100.   0.]

Histogram2d
"""""""""""

The :ref:`Histogram2d <Histogram2d>` is 2-dimensional version of a histogram. It holds the 2-dimensional array and its
datatype has two sets of bin edges.

.. literalinclude:: ../../../macro/tutorial/basic/03_hist2d.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 14,19,22-24,26
    :caption: :download:`03_hist2d.py <../../../macro/tutorial/basic/03_hist2d.py>`

The code produces the following output:

.. code-block:: text
    :linenos:

    Output: [out] hist: hist2d, 12x8=96 bins, edges 0.0->12.0 and 0.0->8.0
    DataType: hist2d, 12x8=96 bins, edges 0.0->12.0 and 0.0->8.0
    Bin edges (X): [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    Bin edges (Y): [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    Data: [[      0.    8836.   34596.   76176.  132496.  202500.  285156.  379456.]
     [ 484416.  599076.  722500.  853776.  992016. 1136356. 1285956. 1440000.]
     [1597696. 1758276. 1920996. 2085136. 2250000. 2414916. 2579236. 2742336.]
     [2903616. 3062500. 3218436. 3370896. 3519376. 3663396. 3802500. 3936256.]
     [4064256. 4186116. 4301476. 4410000. 4511376. 4605316. 4691556. 4769856.]
     [4840000. 4901796. 4955076. 4999696. 5035536. 5062500. 5080516. 5089536.]
     [5089536. 5080516. 5062500. 5035536. 4999696. 4955076. 4901796. 4840000.]
     [4769856. 4691556. 4605316. 4511376. 4410000. 4301476. 4186116. 4064256.]
     [3936256. 3802500. 3663396. 3519376. 3370896. 3218436. 3062500. 2903616.]
     [2742336. 2579236. 2414916. 2250000. 2085136. 1920996. 1758276. 1597696.]
     [1440000. 1285956. 1136356.  992016.  853776.  722500.  599076.  484416.]
     [ 379456.  285156.  202500.  132496.   76176.   34596.    8836.       0.]]


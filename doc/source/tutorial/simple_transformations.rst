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
    :emphasize-lines: 7,10
    :caption: :download:`01_points.py <../../../macro/tutorial/basic/01_points.py>`

The code produces the following output:

.. code-block:: text
    :linenos:

    [obj] Points: 1 transformation(s)
         0 [trans] points: 0 input(s), 1 output(s)
             0 [out] points: array 2d, shape 3x4, size  12

    Transformations: ['points']
    Outputs: ['points']

    Output: [out] points: array 2d, shape 3x4, size  12
    DataType: array 2d, shape 3x4, size  12
    Data:
     [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]]

Let us now follow the code in more details. We prepare 2-dimensional array on a side of python:

.. literalinclude:: ../../../macro/tutorial/basic/01_points.py
    :lines: 8

In order to use this data in the computational chain a transformation should be provided. The :ref:`Points`
transformation is used for arrays. We use ``Points`` constructor from ``constructors`` module in order to initialize it
from the numpy array [#]_.

.. literalinclude:: ../../../macro/tutorial/basic/01_points.py
    :lines: 10

Here ``parray`` is ``GNAObject``. We now may print the information about its transformations, inputs and outputs:

.. literalinclude:: ../../../macro/tutorial/basic/01_points.py
    :lines: 13

.. code-block:: text
    :linenos:

    [obj] Points: 1 transformation(s)
         0 [trans] points: 0 input(s), 1 output(s)
             0 [out] points: array 2d, shape 3x4, size  12

As it can be seen from the output, the ``Points`` instance has a single transformation called ``points`` with a single
output again called ``points``. As it was shown in the :ref:`tutorial_introduction` the transformation may be accessed
by its name as an attribute of the object as ``object.transformation_name``:

.. code-block:: python

    t = parray.points
    print(t)

.. code-block:: text

   [trans] points: 0 input(s), 1 output(s)

The short way to access its output is similar, ``object.transformation_name.output_name``. In our case it reads as
follows:

.. code-block:: python

    output = parray.points.points
    print(output)

.. code-block:: text

   [out] points: array 2d, shape 3x4, size  12

There exist a longer but in come cases more readable way of accessing the same data:

.. code-block:: python

    output = parray.transformations['points'].outputs['points']
    print(output)

.. code-block:: text

   [out] points: array 2d, shape 3x4, size  12

Here we read the dictionary of transformations, request transformation `points`, access the dictionary with its outputs
and request the output `points`.

As we now can access the transformation output, we may request the data it holds:

.. code-block:: python

    arr = parray.points.points.data()
    print(arr)
    print('shape:', arr.shape)

.. code-block:: text

   [[ 0.  1.  2.  3.]
    [ 4.  5.  6.  7.]
    [ 8.  9. 10. 11.]]
   shape: (3, 4)

The ``data()`` method triggers the transformation function which does the calculation and returns a numpy view on the
result, contained in ``parray.points.points``. Accessing the ``data()`` for the second time will do one of the following
things:
  - Return the same view on a cached data in case no calculation is required.
  - If some of the prerequisites of the output has changed the transformation function will be called again updating the
    result. The view on the updated data is returned then.

The status of the transformation may be checked by accessing its ``taintflag``:

.. code-block:: python

    print(bool(parray.points.tainted()))

If the result of the method is `false`, the call to ``data()`` will return cached data without triggering the
transformation function. In case it is `true`, the call to ``data()`` will execute the transformation function and then
return the view to updated data [#]_.

The term `view` here means that if the data will be modified by the transformation, the ``arr`` variable will contain
the updated data. In the same time access to ``arr`` does not trigger the calculation itself, only ``data()`` does.

In case user wants to have a fixed version of the data the ``copy()`` method should be used:

.. code-block:: python

    arr = parray.points.points.data().copy()
    print(arr)
    print('shape:', arr.shape)

There is also ``datatype()`` method that returns a ``DataType`` instance holding the information on the array
dimensions.

.. code-block:: python

    dt = parray.points.points.datatype()
    print(dt)

Now we have defined a transformation holding the data. The transformations output may now be connected to other
transformations' inputs in order to build a computational chain (see :ref:`tutorial_compound_transformations`). It is
important to understand that the way to access transformations and their inputs and outputs is universal and is
applicable to any ``GNAObject``.

.. [#] ``ROOT.Points`` constructor may be used directly, but with more complex signature.
.. [#] It worth noting that while in general the description of the taint mechanism is valid, the transformation
       ``Points`` have no inputs and does not depend on any variables. Therefore, after first execution its
       ``taintflag`` will be `false` forever.

Histogram
"""""""""

The :ref:`Histogram` transformation stores a 1-dimensional histogrammed data. It is very similar to the 1d version of
`Points` with the only difference: its DataType stores the bin edges.

.. literalinclude:: ../../../macro/tutorial/basic/02_hist.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 12,13,21
    :caption: :download:`02_hist.py <../../../macro/tutorial/basic/02_hist.py>`

The work flow for a histogram is very similar to the one of the array. The object has a single transformation `hist`
with a single output `hist`.

The main difference is that ``DataType`` of the histogram now has histogram edges defined. On the line 21
`datatype.edges` C++ vector is accessed and converted to to the python list.

The code produces the following output:

.. code-block:: text
    :linenos:

    [obj] Histogram: 1 transformation(s)
         0 [trans] hist: 0 input(s), 1 output(s)
             0 [out] hist: hist,  12 bins, edges 1.0->7.0, width 0.5

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
    :emphasize-lines: 14,15,23,24
    :caption: :download:`03_hist2d.py <../../../macro/tutorial/basic/03_hist2d.py>`

And again the general work flow is very similar. When it comes to the multiple axes their bin edges may be accessed via
``edgesNd`` member of the ``DataType`` by axis index: see lines 23 and 24.

The code produces the following output:

.. code-block:: text
    :linenos:

    [obj] Histogram2d: 1 transformation(s)
         0 [trans] hist: 0 input(s), 1 output(s)
             0 [out] hist: hist2d, 12x8=96 bins, edges 0.0->12.0 and 0.0->8.0

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


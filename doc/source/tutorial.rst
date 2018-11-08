Tutorial (Under construction)
-----------------------------

Introduction
^^^^^^^^^^^^

GNA provides a way to build a numerical model as lazy evaluated computational graph. The nodes of a graph represent
functions and are called transformations. The transformations produce few return arrays, called outputs, and may have
arguments, called inputs. Outputs of transformations may be connected to inputs of other transformations. They all are
represented by the graph edges.

The data is allocated on the transformations' outputs while inputs are simple views on the relevant data.

Transformations may depend on a number of variables.

The main object in GNA is ``GNAObject`` which holds the following information:
    1. A list of transformations, accessible by name or by index.
    2. A list of variables the transformations depend on.

The typical syntax includes:

.. code-block:: python
    :linenos:

    # Create GNAObject holding some transformations
    obj = SomeGNAObject()

    # There are several ways to access the transformation 'tname' from obj
    # 1. By keyword from a dictionary
    trans = obj.transformrations['tname']
    # 2. By attribute (syntactic sugar for 1.)
    trans = obj.transformrations.tname
    # 3. By index from a dictionary
    trans = obj.transformrations[0]
    # 4. Or even more shorter versions or 2.
    trans = obj['tname']
    trans = obj.tname

    # Similar ways are available for accessing transfromations' outputs
    out = trans.outputs['oname']
    out = trans.outputs.oname
    out = trans.outputs[0]
    # The short ways are only valid in case there are no inputs with name 'oname'
    out = trans['oname']
    out = trans.oname

    # Similar ways are available for accessing transfromations' inputs
    inp = trans.inputs['iname']
    inp = trans.inputs.iname
    inp = trans.inputs[0]
    # The short ways are only valid in case there are no outputs with name 'iname'
    inp = trans['iname']
    inp = trans.iname

.. note:: Any ambiguity will trigger the exception. For example, if transformation has input and output with similar
          name `thename`, shortcut :code:`trans.thename` is forbidden.

Let us no see the actual code examples.

Utilites
^^^^^^^^

GNA contains a set of helper tools to simplify calling C++ functions from python. The :ref:`Constructors` module contain
wrappers for often used C++ classes' enabling the user to pass numpy arrays and python lists. The following code
converts the python list of strings to ``std::vector<std::string>``:

.. code-block:: python
    :linenos:

    from __future__ import print_function
    import constructors as C
    vec = C.stdvector( ['str1', 'str2', 'str3'] )
    print(vec, list(vec))

The code produces the following output:

.. code-block:: txt
    :linenos:

    <ROOT.vector<string> object at 0x55b4546e8be0>, ['str1', 'str2', 'str3']


Simple transformations
^^^^^^^^^^^^^^^^^^^^^^

Here we will review several simple transformations. Simple also means that they do not depend on any variables.

Points
""""""

The :ref:`Points <Points>` transformation is used to represent 1d/2d array as transformation output. The ``Points``
instance is created with ``numpy`` array passed as input:

.. code-block:: python
    :linenos:

    from __future__ import print_function
    import constructors as C
    import numpy as N
    # Create numpy array
    narray = N.arange(12).reshape(3,4)
    # Create a points instance with data, stored in `narray`
    parray = C.Points(narray)

    # Import helper library to make print output more informative
    from gna import printing
    # Access the output `points` of transformation `points` of the object `parray`
    print('Output:', parray.points.points)
    # Access and print relevant DataType
    print('DataType:', parray.points.points.datatype())
    # Access the actual data
    print('Data:\n', parray.points.points.data())

The code produces the following output:

.. code-block:: txt
    :linenos:

    Output: [out] points: array 2d, shape 3x4, size  12
    DataType: array 2d, shape 3x4, size  12
    Data:
     [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]]


Histogram
"""""""""

The :ref:`Histogram` transformation stores a 1-dimensional histogrammed data. TBC... 

.. It is very similar

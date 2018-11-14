.. _tutorial_introduction:

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
    trans = obj.transformations['tname']
    # 2. By attribute (syntactic sugar for 1.)
    trans = obj.transformations.tname
    # 3. By index from a dictionary
    trans = obj.transformations[0]
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

Each input and output has methods ``data()`` and ``datatype()``:
  - ``data()`` returns numpy view on the relevant data with proper shape.
  - ``datatype()`` returns ``DataType`` object with information on data shape and bin edges (for histogram).

Let us no see the actual code examples.


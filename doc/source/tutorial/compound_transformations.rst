.. _tutorial_compound_transformations:

Sum and product: transformations with inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now let us switch to the transformations with inputs. Transformation with inputs is in a sense a function with
arguments. We are still working with transformations that do not depend on variables, therefore the result of the
following transformations will be computed only once.

.. _tutorial_sum:

Sum
"""

The :ref:`Sum` transformation is used to do elementwise sum of the arrays and histograms. See the following code:

.. literalinclude:: ../../../macro/tutorial/compound/01_compound_sum.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 10, 14
    :caption: :download:`01_compound_sum.py <../../../macro/tutorial/compound/01_compound_sum.py>`

The ``Sum`` object is created without arguments:

.. literalinclude:: ../../../macro/tutorial/compound/01_compound_sum.py
    :lines: 13

It may have any number of inputs. Each input is binded with method ``add(output)``, as it is done in:

.. literalinclude:: ../../../macro/tutorial/compound/01_compound_sum.py
    :lines: 17

The printout of the ``Sum`` instance now contains list of all bound inputs:

.. code-block:: text

   [obj] Sum: 1 transformation(s)
        0 [trans] sum: 5 input(s), 1 output(s)
            0 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            1 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            2 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            3 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            4 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            0 [out] sum: array 2d, shape 3x4, size  12

The outputs should be of the same type and shape. The code produces the following output:

.. code-block:: text

   Input 0:
   [[ 0.  1.  2.  3.]
    [ 4.  5.  6.  7.]
    [ 8.  9. 10. 11.]]

   Input 1:
   [[ 1.  2.  3.  4.]
    [ 5.  6.  7.  8.]
    [ 9. 10. 11. 12.]]

   Input 2:
   [[ 2.  3.  4.  5.]
    [ 6.  7.  8.  9.]
    [10. 11. 12. 13.]]

   Input 3:
   [[ 3.  4.  5.  6.]
    [ 7.  8.  9. 10.]
    [11. 12. 13. 14.]]

   Input 4:
   [[ 4.  5.  6.  7.]
    [ 8.  9. 10. 11.]
    [12. 13. 14. 15.]]

   [obj] Sum: 1 transformation(s)
        0 [trans] sum: 5 input(s), 1 output(s)
            0 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            1 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            2 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            3 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            4 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            0 [out] sum: array 2d, shape 3x4, size  12

   Sum result:
   [[10. 15. 20. 25.]
    [30. 35. 40. 45.]
    [50. 55. 60. 65.]]

Ways to construct Sum and add inputs
"""""""""""""""""""""""""""""""""""

There several ways to add inputs to the ``Sum`` instance:

    #. Construct ``Sum`` passing list of outputs as an argument.
    #. Construct empty ``Sum`` and add each output manually via ``add()`` method.
    #. Construct empty ``Sum``, create a set of named inputs via ``add_input()`` method and bind them manually.

The second method was shown in the example above:

.. literalinclude:: ../../../macro/tutorial/compound/01_compound_sum.py
    :lines: 17

All of the methods in comparison are presented in the following example:

.. literalinclude:: ../../../macro/tutorial/compound/02_sum_variants.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 10, 16, 18, 19
    :caption: :download:`02_sum_variants.py <../../../macro/tutorial/compound/02_sum_variants.py>`

First method is shown on line 10: the ``Sum`` constructor from ``constructors`` module may have a list of outputs as an
argument.

.. literalinclude:: ../../../macro/tutorial/compound/02_sum_variants.py
    :lines: 13

The second method, the one shown in the previous example, is represented by lines 11 and 16:

.. literalinclude:: ../../../macro/tutorial/compound/02_sum_variants.py
    :lines: 14, 17-19

The third method is very close to the second one. It is represented by lines 12, 18 and 19:

.. literalinclude:: ../../../macro/tutorial/compound/02_sum_variants.py
    :lines: 15, 17-18, 21-22

The binding is done in two steps. At first we create a named input:

.. literalinclude:: ../../../macro/tutorial/compound/02_sum_variants.py
    :lines: 21

The input then  is connected via call:

.. literalinclude:: ../../../macro/tutorial/compound/02_sum_variants.py
    :lines: 22

The example script prints the inputs and outputs for each case:

.. code-block:: text

   Sum, configured via constructor
   [obj] Sum: 1 transformation(s)
        0 [trans] sum: 5 input(s), 1 output(s)
            0 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            1 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            2 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            3 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            4 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            0 [out] sum: array 2d, shape 3x4, size  12

   Sum, configured via add() method
   [obj] Sum: 1 transformation(s)
        0 [trans] sum: 5 input(s), 1 output(s)
            0 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            1 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            2 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            3 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            4 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            0 [out] sum: array 2d, shape 3x4, size  12

   Sum, configured via add_input() method
   [obj] Sum: 1 transformation(s)
        0 [trans] sum: 5 input(s), 1 output(s)
            0 [in]  input_00 -> [out] points: array 2d, shape 3x4, size  12
            1 [in]  input_01 -> [out] points: array 2d, shape 3x4, size  12
            2 [in]  input_02 -> [out] points: array 2d, shape 3x4, size  12
            3 [in]  input_03 -> [out] points: array 2d, shape 3x4, size  12
            4 [in]  input_04 -> [out] points: array 2d, shape 3x4, size  12
            0 [out] sum: array 2d, shape 3x4, size  12

   Results:
   [[10. 15. 20. 25.]
    [30. 35. 40. 45.]
    [50. 55. 60. 65.]]

   [[10. 15. 20. 25.]
    [30. 35. 40. 45.]
    [50. 55. 60. 65.]]

   [[10. 15. 20. 25.]
    [30. 35. 40. 45.]
    [50. 55. 60. 65.]]

See that in case three the names of the inputs are defined as user has specified.

.. _tutorial_product:

Product
"""""""

The ``Product`` transformation is very similar to the ``Sum``. The example, one to one similar to the example from the
previous section may be found below. The main differences are that there is a method ``multiply(output)`` for binding
outputs. The object defines transformation `product` with output, named `prodcut`.

.. literalinclude:: ../../../macro/tutorial/compound/03_product_variants.py
    :linenos:
    :lines: 4-
    :caption: :download:`03_product_variants.py <../../../macro/tutorial/compound/03_product_variants.py>`

The script produces the following output. Check it and compare with the output of the ``Sum`` example from the previous
section.

.. code-block:: text

   Product, configured via constructor
   [obj] Product: 1 transformation(s)
        0 [trans] product: 5 input(s), 1 output(s)
            0 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            1 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            2 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            3 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            4 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            0 [out] product: array 2d, shape 3x4, size  12

   Product, configured via multiply() method
   [obj] Product: 1 transformation(s)
        0 [trans] product: 5 input(s), 1 output(s)
            0 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            1 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            2 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            3 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            4 [in]  points -> [out] points: array 2d, shape 3x4, size  12
            0 [out] product: array 2d, shape 3x4, size  12

   Product, configured via add_input() method
   [obj] Product: 1 transformation(s)
        0 [trans] product: 5 input(s), 1 output(s)
            0 [in]  input_00 -> [out] points: array 2d, shape 3x4, size  12
            1 [in]  input_01 -> [out] points: array 2d, shape 3x4, size  12
            2 [in]  input_02 -> [out] points: array 2d, shape 3x4, size  12
            3 [in]  input_03 -> [out] points: array 2d, shape 3x4, size  12
            4 [in]  input_04 -> [out] points: array 2d, shape 3x4, size  12
            0 [out] product: array 2d, shape 3x4, size  12

   Results:
   [[0.0000e+00 1.2000e+02 7.2000e+02 2.5200e+03]
    [6.7200e+03 1.5120e+04 3.0240e+04 5.5440e+04]
    [9.5040e+04 1.5444e+05 2.4024e+05 3.6036e+05]]

   [[0.0000e+00 1.2000e+02 7.2000e+02 2.5200e+03]
    [6.7200e+03 1.5120e+04 3.0240e+04 5.5440e+04]
    [9.5040e+04 1.5444e+05 2.4024e+05 3.6036e+05]]

   [[0.0000e+00 1.2000e+02 7.2000e+02 2.5200e+03]
    [6.7200e+03 1.5120e+04 3.0240e+04 5.5440e+04]
    [9.5040e+04 1.5444e+05 2.4024e+05 3.6036e+05]]

.. _tutorial_binding_syntax:

Simplified syntax for working with inputs and outputs
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Now let us review the python syntax used to bind transformations together. Let us start with an array with one element:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 4-9
    :caption: :download:`04_syntax.py <../../../macro/tutorial/compound/04_syntax.py>`

By default points is an object with one transformation and one output, which may be accessed as follows:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 11-13

The method ``single()`` may be used to return the single output of a single object:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 14

Which is equivalent to:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 13

Transformation also has a method ``single()`` which returns the single output of a transformation. Both methods raise an
exception if single output can not be returned or the output is not single.

The default way to connect the output to the input is to use ``input.connect(output)`` method:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 16-19

`Single` notation may also be used in a similar way:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 21-24

Here the method ``single_input()`` returns a single input of a single transformation.

There exist robust operators ``>>`` and ``<<`` that may be used to connect output to input as ``output>>input``:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 26-29

or as ``input<<output``:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 31-34

Operators ``<<`` and ``>>`` may take care in cases when there is a single input/output in the transformation:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 36-39

or when there is a single transformation in the object:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 41-44

The short version may go either on the left hand side, rite hand side or both sides of the expression.

More then one inputs may be passed. In this case the output is connected to each of the inputs, like this:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 46-50

or like this:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 52-56

In the above example we created a chain that passes as single value transformation to transformation. In the last two
examples we have doubled the value twice. Therefore the output should contain the value four times larger:

.. literalinclude:: ../../../macro/tutorial/compound/04_syntax.py
    :linenos:
    :lines: 57-

Which is confirmed by the output:

.. code-block:: text

   Expect: 4.938
   Got:    4.938

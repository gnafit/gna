.. _tutorial_compound_transformations:

Compound transformaions
^^^^^^^^^^^^^^^^^^^^^^^

Now let us switch to the transformations with inputs. Transformation with inputs is in a sense a function with
arguments. We are still working with transformations that do not depend on variables, therefore the result of the
following transformations will be computed only once.

Sum
"""

The :ref:`Sum` transformation is used to do elementwise sum of the arrays and histograms. See the following code:

.. literalinclude:: ../../../macro/tutorial/compound/01_compound_sum.py
    :linenos:
    :lines: 4-
    :emphasize-lines: 10, 14
    :caption: :download:`01_compound_sum.py <../../../macro/tutorial/basic/01_compound_sum.py>`

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

Ways to costruct Sum and add inputs
"""""""""""""""""""""""""""""""""""

There several ways to add inputs to the ``Sum`` instance. One of the was shown in the previous example. Each element of
the sum is added directly via ``add(output)`` method.

.. literalinclude:: ../../../macro/tutorial/compound/01_compound_sum.py
    :lines: 17


Product
"""""""


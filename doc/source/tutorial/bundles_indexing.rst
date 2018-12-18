Indexing
""""""""

Multidimensional indices enable the user to define on how to replicate transformations and variables. The main
operations for indices are:

#. Iterate over all possible combinations of the values of the indices.
#. Get the current values of indices.
#. Format strings using current values of indices.
#. Get a subset of indices.
#. Merge few subsets of indices.

0d case
'''''''

Let us start from the simplest possible example of empty index:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 4-18
    :caption: :download:`01_indexing.py <../../../macro/tutorial/bundles/01_indexing.py>`

The index is created by instantiating an ``NIndex`` class:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :lines: 10

The ``NIndex`` instance is ready for iteration:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :lines: 13-18

A copy of ``NIndex`` is returned on each iteration with `current values` set. A `current_values()` method returns a
tuple with all current index values, which is empty in our case. A `current_format()` method formats a string using the
current values. If a name is provided as an argument it is also added to the format.

The following output is produced. As one may see, empty index still produces a single iteration.

.. code-block:: text

    Test 0d index
      iteration 0
        index:     <empty string>
        full name: var
        values:    ()

1d case
'''''''

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 20-34
    :caption: :download:`01_indexing.py <../../../macro/tutorial/bundles/01_indexing.py>`

.. code-block:: text

    Test 1d index
      iteration 0
        index:     1
        full name: var.1
        values:    ('1',)

      iteration 1
        index:     2
        full name: var.2
        values:    ('2',)

      iteration 2
        index:     3
        full name: var.3
        values:    ('3',)

2d case
'''''''

.. attention::

    The following content will be documented soon.


.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 1-
    :caption: :download:`01_indexing.py <../../../macro/tutorial/bundles/01_indexing.py>`

.. code-block:: text

    Test 2d index
      iteration 0
        index:     1.a
        full name: var.1.a
        values:    ('1', 'a')

      iteration 1
        index:     1.b
        full name: var.1.b
        values:    ('1', 'b')

      iteration 2
        index:     1.c
        full name: var.1.c
        values:    ('1', 'c')

      iteration 3
        index:     2.a
        full name: var.2.a
        values:    ('2', 'a')

      iteration 4
        index:     2.b
        full name: var.2.b
        values:    ('2', 'b')

      iteration 5
        index:     2.c
        full name: var.2.c
        values:    ('2', 'c')

      iteration 6
        index:     3.a
        full name: var.3.a
        values:    ('3', 'a')

      iteration 7
        index:     3.b
        full name: var.3.b
        values:    ('3', 'b')

      iteration 8
        index:     3.c
        full name: var.3.c
        values:    ('3', 'c')

    Test 3d index and arbitrary name position
      iteration 0
        full name: clone_00.var.SA.D1
        values:    ('clone_00', 'var', 'SA', 'D1')

      iteration 1
        full name: clone_00.var.SA.D2
        values:    ('clone_00', 'var', 'SA', 'D2')

      iteration 2
        full name: clone_00.var.SB.D1
        values:    ('clone_00', 'var', 'SB', 'D1')

      iteration 3
        full name: clone_00.var.SB.D2
        values:    ('clone_00', 'var', 'SB', 'D2')

      iteration 4
        full name: clone_01.var.SA.D1
        values:    ('clone_01', 'var', 'SA', 'D1')

      iteration 5
        full name: clone_01.var.SA.D2
        values:    ('clone_01', 'var', 'SA', 'D2')

      iteration 6
        full name: clone_01.var.SB.D1
        values:    ('clone_01', 'var', 'SB', 'D1')

      iteration 7
        full name: clone_01.var.SB.D2
        values:    ('clone_01', 'var', 'SB', 'D2')

    Test 4d index and separated iteration
      major iteration 0
        major values:    ('SA', 'D1')
          minor iteration 0
            minor values:    ('clone_00', 'e1')
            full name: clone_00.var.SA.D1.e1
            custom label: Flux from SA to D1 element e1 (clone_00)
          minor iteration 1
            minor values:    ('clone_00', 'e2')
            full name: clone_00.var.SA.D1.e2
            custom label: Flux from SA to D1 element e2 (clone_00)
          minor iteration 2
            minor values:    ('clone_00', 'e3')
            full name: clone_00.var.SA.D1.e3
            custom label: Flux from SA to D1 element e3 (clone_00)
          minor iteration 3
            minor values:    ('clone_01', 'e1')
            full name: clone_01.var.SA.D1.e1
            custom label: Flux from SA to D1 element e1 (clone_01)
          minor iteration 4
            minor values:    ('clone_01', 'e2')
            full name: clone_01.var.SA.D1.e2
            custom label: Flux from SA to D1 element e2 (clone_01)
          minor iteration 5
            minor values:    ('clone_01', 'e3')
            full name: clone_01.var.SA.D1.e3
            custom label: Flux from SA to D1 element e3 (clone_01)

    Test 4d index and separated iteration
      major iteration 0
        major values:    ('D1', 'g1')
            full values: ('D1', 'SA', 'g1')
            full values: ('D1', 'SB', 'g1')

      major iteration 1
        major values:    ('D1', 'g2')
            full values: ('D1', 'SA', 'g2')
            full values: ('D1', 'SB', 'g2')

      major iteration 2
        major values:    ('D2', 'g1')
            full values: ('D2', 'SA', 'g1')
            full values: ('D2', 'SB', 'g1')

      major iteration 3
        major values:    ('D2', 'g2')
            full values: ('D2', 'SA', 'g2')
            full values: ('D2', 'SB', 'g2')

    Test 4d index and separated iteration: try to mix dependent indices
      major iteration 0
        major values:    ('D1', 'e1')
            full values: ('D1', 'SA', 'e1')
            full values: ('D1', 'SB', 'e1')

      major iteration 1
        major values:    ('D1', 'e2')
            full values: ('D1', 'SA', 'e2')
            full values: ('D1', 'SB', 'e2')

      major iteration 2
        major values:    ('D1', 'e3')
            full values: ('D1', 'SA', 'e3')
            full values: ('D1', 'SB', 'e3')

      major iteration 3
        major values:    ('D2', 'e1')
            full values: ('D2', 'SA', 'e1')
            full values: ('D2', 'SB', 'e1')

      major iteration 4
        major values:    ('D2', 'e2')
            full values: ('D2', 'SA', 'e2')
            full values: ('D2', 'SB', 'e2')

      major iteration 5
        major values:    ('D2', 'e3')
            full values: ('D2', 'SA', 'e3')
            full values: ('D2', 'SB', 'e3')


.. _tutorial_indexing:

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

Now let us create an 1d index: Do do this we add a definition of one of the indices:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 23-25

Typical definition contains three items: short name, long name and a list with values. Short name will be used for
reference. Long name is used for string formatting.

When iterated,

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 27-29

the resulting index will contain the value from the list on each iteration:

.. code-block:: text

  iteration 0:     values:    ('1',)
  iteration 1:     values:    ('2',)
  iteration 2:     values:    ('3',)

The `current_format()` returns a single string for each iteration.

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 34-36

For 1d case the string will be equal to each of the values:

.. code-block:: text

  iteration 0:     index:     1
  iteration 1:     index:     2
  iteration 2:     index:     3

When `name` field is added to the `current_format()` call, name is joined to the values of indices with `.` as a
separator:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 39-41

.. code-block:: text

  iteration 0:     full name: var.1
  iteration 1:     full name: var.2
  iteration 2:     full name: var.3


2d case
'''''''

In order to make 2d multi index just add another index specification:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 47-50

The iteration of the 2d index

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 53-57

will produce all possible pairs of the values of `i` and `j` indices:

.. code-block:: text

   iteration 0
     index:     1.a
     full name: var.1.a
     values:    ('1', 'a')

   iteration 1
     index:     1.b
     full name: var.1.b
     values:    ('1', 'b')

   iteration 2
     index:     2.a
     full name: var.2.a
     values:    ('2', 'a')

   iteration 3
     index:     2.b
     full name: var.2.b
     values:    ('2', 'b')

   iteration 4
     index:     3.a
     full name: var.3.a
     values:    ('3', 'a')

   iteration 5
     index:     3.b
     full name: var.3.b
     values:    ('3', 'b')

The order of the indices is kept to the order of the initialization.

3d case and name position
'''''''''''''''''''''''''

For 3d case we will introduce the indices `z`, `s` and `d`. Where `s` is used for sources, `d`. Consider the case when
we want to create several clones of the same model, the `z` will be used to index them. In this case we also would like
so that the variables and outputs of the same models were stored together. This may be achieved by putting the name
after the clone index:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 64-69

When the string `name` is passed instead of index definition, the position of the `name` will be used to position the
actual name.

Now iterating `nidx` will produce all possible combinations of the `z`, `s` and `d` indices in the order they were
specified:

.. code-block:: text

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

Also, note that the name in this example goes after the clone index.

4d indexing and partial iteration
'''''''''''''''''''''''''''''''''

Let us now look at 4d case.

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 81-87

It is often needed to iterate only a part of the indices of a multidimensional index. This may be achieved by
`split(short_names)` method:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 89

It produces a pair of ``NIndex`` instances. First one contains the indices from the list of short names, passed as
argument. In this case it contains `s` and `d`. The second ``NIndex`` contains all the other indices: `z` and `e` in
this case.

These two instances may be iterated independently:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 90-96

This kind of iteration is useful, for example, in the cases, when variables depend only on a part of the indices.

The partially iterated indices may be combined together:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 98-99

The `current_format()` method has an optional argument: python format specification. In this case the string will be
formatted using the values of the current indices, which may be referenced by both short and long names:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 100

The format string may reference any other positional and keyword arguments which should be passed to the
`current_format()` as well.

The output for the example with 4d index is the following:

.. code-block:: text

   major iteration 0
     major values:    ('SA', 'D1')
       minor iteration 0
         minor values:   ('clone_00', 'e1')
         full name:      clone_00.var.SA.D1.e1
         custom label:   Flux from SA to D1 element e1 (clone_00)
       minor iteration 1
         minor values:   ('clone_00', 'e2')
         full name:      clone_00.var.SA.D1.e2
         custom label:   Flux from SA to D1 element e2 (clone_00)
       minor iteration 2
         minor values:   ('clone_00', 'e3')
         full name:      clone_00.var.SA.D1.e3
         custom label:   Flux from SA to D1 element e3 (clone_00)
       minor iteration 3
         minor values:   ('clone_01', 'e1')
         full name:      clone_01.var.SA.D1.e1
         custom label:   Flux from SA to D1 element e1 (clone_01)
       minor iteration 4
         minor values:   ('clone_01', 'e2')
         full name:      clone_01.var.SA.D1.e2
         custom label:   Flux from SA to D1 element e2 (clone_01)
       minor iteration 5
         minor values:   ('clone_01', 'e3')
         full name:      clone_01.var.SA.D1.e3
         custom label:   Flux from SA to D1 element e3 (clone_01)

We stop after the first major iteration to keep the output short.

Dependant indices
'''''''''''''''''

Sometimes it is useful to have several indices for the same thing. For example, imagine the detector elements come in
groups and may be referenced by both element number or group number. The group index then may be used to specify some
common variables for the group.

This situation may be handled by the ``NIndex`` as well. The resolution rules are the following:

#. The group indices are called `slave` indices, while the elements indices within groups are called `master` indices.
#. When we work with a set of indices, that contains a `slave` index but not its master, the `slave` index values
   will be iterated.
#. In any case when `slave` index is combined with its `master`, only `master` index will be iterated. In the same time
   the relevant `slave` index value may be obtained for each `master` value on each iteration.

To work with dependant indices we need to tell the `master` index how its values are grouped:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 108-113

In our case for `e` index we have provided a dictionary with short and long names of the slave index, as well as the
list of pairs `slave index` and `master indices`. In our example group `g1` contains elements `e1` and `e2`, the group
`g2` contains the only element `e3`.

Now let us look how this work. Let us split the multidimensional index in two:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 116-117

The first group contains indices `d` and `g`, while the second contains the index `s`. Index `e` is not used since
`g` was required. We have also requested multi-index with `e` index for later use. The iteration over `nidx_major` and
`nidx_minor` produces the following output:

.. code-block:: text

   major iteration 0
     major values:    ('D1', 'g1')
       full values 0: ('D1', 'SA', 'g1')
       full values 1: ('D1', 'SB', 'g1')

   major iteration 1
     major values:    ('D1', 'g2')
       full values 0: ('D1', 'SA', 'g2')
       full values 1: ('D1', 'SB', 'g2')

   major iteration 2
     major values:    ('D2', 'g1')
       full values 0: ('D2', 'SA', 'g1')
       full values 1: ('D2', 'SB', 'g1')

   major iteration 3
     major values:    ('D2', 'g2')
       full values 0: ('D2', 'SA', 'g2')
       full values 1: ('D2', 'SB', 'g2')

As one may see, there is each combination of `d`, `s` and `g` values.

Now let us combine `nidx_major` with `nidx_e`. The former one contains index `g` while the latter one contains index
`e`, which may not be combined together.

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 129

The result will contain only index `e`, but the relevant `g` value may be obtained by specifying the format string:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 137

Here is the output:

.. code-block:: text

  major iteration 0
    major values:    ('D1', 'e1')
      full values 0:      ('D1', 'SA', 'e1')
      formatted string 0: Element e1 in group g1
      full values 1:      ('D1', 'SB', 'e1')
      formatted string 1: Element e1 in group g1

  major iteration 1
    major values:    ('D1', 'e2')
      full values 0:      ('D1', 'SA', 'e2')
      formatted string 0: Element e2 in group g1
      full values 1:      ('D1', 'SB', 'e2')
      formatted string 1: Element e2 in group g1

  major iteration 2
    major values:    ('D1', 'e3')
      full values 0:      ('D1', 'SA', 'e3')
      formatted string 0: Element e3 in group g2
      full values 1:      ('D1', 'SB', 'e3')
      formatted string 1: Element e3 in group g2

  major iteration 3
    major values:    ('D2', 'e1')
      full values 0:      ('D2', 'SA', 'e1')
      formatted string 0: Element e1 in group g1
      full values 1:      ('D2', 'SB', 'e1')
      formatted string 1: Element e1 in group g1

  major iteration 4
    major values:    ('D2', 'e2')
      full values 0:      ('D2', 'SA', 'e2')
      formatted string 0: Element e2 in group g1
      full values 1:      ('D2', 'SB', 'e2')
      formatted string 1: Element e2 in group g1

  major iteration 5
    major values:    ('D2', 'e3')
      full values 0:      ('D2', 'SA', 'e3')
      formatted string 0: Element e3 in group g2
      full values 1:      ('D2', 'SB', 'e3')
      formatted string 1: Element e3 in group g2

The full example
''''''''''''''''

The complete code for this example may be found below:

.. literalinclude:: ../../../macro/tutorial/bundles/01_indexing.py
    :linenos:
    :lines: 4-
    :caption: :download:`01_indexing.py <../../../macro/tutorial/bundles/01_indexing.py>`


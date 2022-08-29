.. _tutorial_parameters_replication:

Replicating parameters
''''''''''''''''''''''

The previous example with reading a set of parameters from the dictionary does not do itself too much functionality. Let us
now use the feature of multi dimensional indexing, described in :ref:`tutorial_indexing`.

The bundle specification
++++++++++++++++++++++++

The `bundle` field of the configuration may contain two options: `nidx` and `major`. The latter is either ``NIndex``
instance or ``NIndex`` specification. `nidx` will provide information on actual computational chain multiplicity. The
former specifies a subset of `nidx` indices that should be considered as major.

It is supposed that bundle will create some distinct configuration for each `nidx_major`, while `nidx_minor` define
exact replicas of each `nidx_major` state.

When applied to the parameters, there should be provided a value with uncertainties for each `nidx_major` iteration. The
parameter will be replicated for each `nidx_minor` iteration.

We expect that the configuration will contain the name of the parameter (`parameter`), a set of the values and
uncertainties (`pars`) and, optionally, a format string for parameter labels (`label`). The code of the bundle reads as
follows:

.. literalinclude:: ../../../macro/tutorial/bundles/parameters_ex02.py
    :linenos:
    :lines: 3-
    :caption: :download:`parameters_ex02.py <../../../macro/tutorial/bundles/parameters_ex02.py>`

.. note::

    The `parameters_ex02` bundle from tutorial is a simplified copy of the `parameters_v01` bundle, provided by GNA.

We start iterating over major indices and for each iteration request a value from the `pars`:

.. literalinclude:: ../../../macro/tutorial/bundles/parameters_ex02.py
    :linenos:
    :lines: 12-17
    :emphasize-lines: 1,2,4

In a special case of 0 dimensions, `current_values()` return empty tuple. In this case we expect `pars` to contain a
single parameter specification.

The next few lines

.. literalinclude:: ../../../macro/tutorial/bundles/parameters_ex02.py
    :linenos:
    :lines: 19-20

implement the loop over minor indices. We create a combined index `it` and use it to produce a formatted label.

And as the last action we define a parameter:

.. literalinclude:: ../../../macro/tutorial/bundles/parameters_ex02.py
    :linenos:
    :lines: 23

The `reqparameter()` method is similar to the environments `reqparameter()`, described :ref:`previously
<tutorial_parameters_def>`. The main difference as it takes current iteration of multi index to properly format the name.

Note, also, that we are using `reqparameter()` instead of `defparameter()`. This method enables the user to predefine
the parameter by the user before executing the bundle.

Let us now check three examples.

Only major indices
++++++++++++++++++

The first one has the following configuration:

.. literalinclude:: ../../../macro/tutorial/bundles/04_bundle_parameters_multiple.py
    :linenos:
    :lines: 11-45
    :caption: :download:`04_bundle_parameters_multiple.py <../../../macro/tutorial/bundles/04_bundle_parameters_multiple.py>`

Now in the `bundle` configuration we have specified indices:

.. literalinclude:: ../../../macro/tutorial/bundles/04_bundle_parameters_multiple.py
    :linenos:
    :lines: 18-22

We have also defined the order of the keys to be `{source}.{name}.{detector}`.

The bundle will create parameters with name `rate0` and the following label:

.. literalinclude:: ../../../macro/tutorial/bundles/04_bundle_parameters_multiple.py
    :linenos:
    :lines: 25,27

The parameters' values and uncertainties are specified further:

.. literalinclude:: ../../../macro/tutorial/bundles/04_bundle_parameters_multiple.py
    :linenos:
    :lines: 29-42

As soon as have taken out `uncertainty` and `mode` specification, all the parameters will have the uncertainty of 1%.

Let us now execute the bundle and print the namespace contents.

.. literalinclude:: ../../../macro/tutorial/bundles/04_bundle_parameters_multiple.py
    :linenos:
    :lines: 44,45

Note, that we have specified, that the bundle used predefined namespace instead of the global one. And the output is
here:

.. code-block:: text

   Variables in namespace 'bundle1.SA.rate0':
     D1                   =          1 │           1±        0.01 [          1%] │ Flux normalization SA->D1
     D2                   =          3 │           3±        0.03 [          1%] │ Flux normalization SA->D2
     D3                   =          5 │           5±        0.05 [          1%] │ Flux normalization SA->D3
   Variables in namespace 'bundle1.SB.rate0':
     D1                   =          2 │           2±        0.02 [          1%] │ Flux normalization SB->D1
     D2                   =          4 │           4±        0.04 [          1%] │ Flux normalization SB->D2
     D3                   =          6 │           6±        0.06 [          1%] │ Flux normalization SB->D3

Major and minor indices
+++++++++++++++++++++++

Let us now slightly tweak the previous example by adding an extra index and setting it to be minor.

.. literalinclude:: ../../../macro/tutorial/bundles/04_bundle_parameters_multiple.py
    :linenos:
    :lines: 50-75
    :emphasize-lines: 7,9,12

We have also provided a special format for label to include new index. As one can see, each of the 6 permutations of the
values of the major index now contains two copies for each of the values of index `e`.

.. code-block:: text

   Variables in namespace 'bundle2.rate1.SA.D1':
     e0                   =          1 │           1±        0.01 [          1%] │ Flux normalization SA->D1 (e0)
     e1                   =          1 │           1±        0.01 [          1%] │ Flux normalization SA->D1 (e1)
   Variables in namespace 'bundle2.rate1.SA.D2':
     e0                   =          3 │           3±        0.09 [          3%] │ Flux normalization SA->D2 (e0)
     e1                   =          3 │           3±        0.09 [          3%] │ Flux normalization SA->D2 (e1)
   Variables in namespace 'bundle2.rate1.SA.D3':
     e0                   =          5 │           5±        0.25 [          5%] │ Flux normalization SA->D3 (e0)
     e1                   =          5 │           5±        0.25 [          5%] │ Flux normalization SA->D3 (e1)
   Variables in namespace 'bundle2.rate1.SB.D1':
     e0                   =          2 │           2±        0.04 [          2%] │ Flux normalization SB->D1 (e0)
     e1                   =          2 │           2±        0.04 [          2%] │ Flux normalization SB->D1 (e1)
   Variables in namespace 'bundle2.rate1.SB.D2':
     e0                   =          4 │           4±        0.16 [          4%] │ Flux normalization SB->D2 (e0)
     e1                   =          4 │           4±        0.16 [          4%] │ Flux normalization SB->D2 (e1)
   Variables in namespace 'bundle2.rate1.SB.D3':
     e0                   =          6 │           6±        0.36 [          6%] │ Flux normalization SB->D3 (e0)
     e1                   =          6 │           6±        0.36 [          6%] │ Flux normalization SB->D3 (e1)


Sanity check: 0d case
+++++++++++++++++++++

Let us double check the setup by passing the 0d index.

.. literalinclude:: ../../../macro/tutorial/bundles/04_bundle_parameters_multiple.py
    :linenos:
    :lines: 82-
    :emphasize-lines: 9

As we have printed the whole namespace it not contains the parameters from all the examples in their own namespaces. The
last line represents the 0d case as defined by the configuration:

.. code-block:: text

   Variables in namespace 'bundle3':
     constant             =         -1 │          -1±       -0.04 [          4%] │ some constant

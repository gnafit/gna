Making parameters via bundles
'''''''''''''''''''''''''''''

Let us start from a bundle that simply initializes some parameters from the list. It is fairly simple:

.. literalinclude:: ../../../macro/tutorial/bundles/parameters_ex01.py
    :linenos:
    :lines: 3-
    :caption: :download:`parameters_ex01.py <../../../macro/tutorial/bundles/parameters_ex01.py>`

The `parameters_ex01` bundle class contains two methods `__init__()` and `define_variables()`. The constructor

.. literalinclude:: ../../../macro/tutorial/bundles/parameters_ex01.py
    :linenos:
    :lines: 6-8

passes all the arguments to the common bundle costructor, which initializes configuration, multidimensional index, etc.
As this particular bundle can not handle indexing, it checks that the number of dimensions is within limits :math:`[0,0]`.

The second method is called to initialize parameters. For each `name, par` pair from the configuration it initializes
a parameter.

.. literalinclude:: ../../../macro/tutorial/bundles/parameters_ex01.py
    :linenos:
    :lines: 10-

Again `parameters_ex01` in a sense a function, which reads its argument from configuration dictionary.

In order to use the bundle one needs to execute its configuration. See the following example:

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :linenos:
    :lines: 4-
    :caption: :download:`03_bundle_parameters.py <../../../macro/tutorial/bundles/03_bundle_parameters.py>`

We start from making a dictionary with configuration:

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :linenos:
    :lines: 11-25

The `bundle` specifies the bundle that should read the configuration and make parameters. The only other field is a
dictionary with parameter specifications. We use `uncertaindict()` for this. Its signature is similar to the python
`dict()` `signature <https://docs.python.org/2/tutorial/datastructures.html#dictionaries>`_: first argument is a list
with (key, value) pairs, other arguments are named and represent `key=value` pairs as well.

The possible variants of parameter definitions should be readable. First line

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :lines: 18

defines parameter `par_a` with central value of 1 and relative uncertainty of 1%, expressed in percents.

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :lines: 19

defines parameter `par_b` with central value of 2 and relative uncertainty of 1%, expressed in relative units.

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :lines: 20

defines parameter `par_c` with central value of 3 and absolute uncertainty of 0.5.

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :lines: 21

defines parameter `a` in a nested namespace `group` with default value of 1. The parameter is free.

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :lines: 22

defines parameter `b` in a nested namespace `group` with value of 1. The parameter is fixed.

Extra argument may be added to each line defining the parameters label as it is done in the last example.

Finally we execute the bundle configuration as follows:

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :lines: 30

The function returns `parameters_ex01` class instance. After execution the global namespace contains all the parameters
defined:

.. literalinclude:: ../../../macro/tutorial/bundles/03_bundle_parameters.py
    :lines: 35

.. code-block:: text

    Variables in namespace '':
      par_a                =          1 │           1±        0.01 [          1%] │
      par_b                =          2 │           2±        0.02 [          1%] │
      par_c                =          3 │           3±         0.5 [    16.6667%] │
    Variables in namespace 'group':
      a                    =          1 │           1±         inf [free]         │
      b                    =          1 │                 [fixed]                 │ Labeled fixed parameter

Such an approach enables the user to detach the configuration from the actual code. In the same time the piece of
configuration indicates which code should handle it. The code of the `parameters_ex01` is meant to be immutable for
backwards compatibility. Alternative behaviour should be achieved by introducing other versions of the bundle or other
bundles.


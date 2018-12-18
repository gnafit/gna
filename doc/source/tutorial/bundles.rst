Increasing complexity
^^^^^^^^^^^^^^^^^^^^^

Up to now we were discussing simple computational chains consisting of few transformations and depending on few
variables. Building a large model may require much more code and typing. In this section we will discuss how to scale the
numerical models up.

The following items will be reviewed:

Multi-dimensional indices.
    It is often necessary to replicate parts of the computational graph. This is required to provide multiple similar
    sources, several detectors, few components in a sum, etc. Multidimensional iterable indices provide the information on
    how much replications should be done and enable GNA to generate proper names and labels.

Bundles.
    Bundle is a small computational chain with open inputs and outputs. Bundles are handled by python classes. Each
    bundle may read configuration dictionary, a multi-dimensional index and build a computational graph, properly
    replicating the instances according to the index.

Nested dictionaries.
    Nested dictionary is an extension of the regular python dictionaries with ability to create nested dictionaries and
    manage multidimensional keys. Nested dictionaries are used to bookkeep the inputs and outputs created within
    bundles.

Expressions.
    Mathematical expressions are used to define the connections between computational graphs created by various bundles.
    As expressions support multidimensional indexing they enable the user to create large scale models.

.. toctree::

    bundles_indexing
    bundles_dictionary
    bundles_bundles
    bundles_expressions


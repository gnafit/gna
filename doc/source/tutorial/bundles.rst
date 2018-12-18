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
    Bundle

.. toctree::

    bundles_indexing
    bundles_dictionary
    bundles_bundles
    bundles_expressions


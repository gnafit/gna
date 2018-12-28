Bundles and replication
"""""""""""""""""""""""

A bundles were created in order to facilitate making small computational graphs and scale them based on a simple
configuration. Here is a short list of bundle capabilities, bundles are able:

  #. Read a configuration dictionary, which contain:

      + Bundle name and version number, telling GNA which bundle to load to read the configuration.
      + Multidimensional index, defining how the bundle should replicate the parameters and/or transformations.
      + Other configuration, specific to a particular bundle.

  #. Access environment and define a set of variables, required by the transformations chain it builds.
  #. Define and bind a set of transformations. Register the open inputs and outputs so other bundles may access them.
  #. Require inputs and outputs, provided by the other transformations and bind them.

Each of the items above is optional: the bundle may only define parameters; or only make a computational graph with some
outputs; or it may make nothing but simply bind inputs and outputs already defined by the other bundles.

.. toctree::

    bundles_intro
    bundles_parameters
    bundles_parameters_replication
    bundles_graphs

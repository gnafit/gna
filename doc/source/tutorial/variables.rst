Variables
^^^^^^^^^

Combining objects that do not change is not very interesting task. The GNA was designed in order to work with models
with large number of parameters. In the simplest case the variables represent weights in a sum of arrays
(:ref:`WeightedSum`). In the more complex case variables represent some physical parameters, like parameters of energy
resolution model (:ref:`EnergyResolution`).

We will start working with variables by introducing the concept of environment and namespaces.

Environment
"""""""""""

Environment is a global GNA object defined in Python and holding a folder like structure with variables.
When ``GNAObject`` is created it requests a list of variables it depends on from the environment. The variables are then
bound to the transformations. As the value of the variable changed the `taintflag=false` is propagated to the dependent
variables and transformations invalidating the cached data.

.. toctree::

   variables_def
   variables_corr
   weightedsum


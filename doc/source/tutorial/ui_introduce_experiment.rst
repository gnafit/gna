Introducing new experiments
"""""""""""""""""""""""""""

From the point of view of GNA command line user interface and experiment is just an UI module, that announces the
outputs within common environment. In order to separate the experiments from regular UI modules they are loaded via a
dedicated UI module `exp`:

.. code-block:: bash

    ./gna -- exp --help

The experiment implementations are expected to be located in on of the `experimentpaths`, defined in
`config/gna/gnacfg.py` file. The default configuration may be overridden, see the `experimentpaths` example from
:ref:`bundles <bundles_configuration>` tutorial.

Let us now add the tutorial path to the `experimentpaths` and define a new experiment.

.. code-block:: bash

    mkdir -p config_local/gna
    echo "experimentpaths = [ './pylib/gna/experiments', './macro/tutorial/ui' ]" >> config_local/gna/gnacfg.py



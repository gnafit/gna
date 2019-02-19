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

Let us now use the tutorial path for our `experimentpaths` and define a new experiment.

.. code-block:: bash

    mkdir -p config_local/gna
    echo "experimentpaths = ['./macro/tutorial/ui/experiments']" >> config_local/gna/gnacfg.py

The command :code:`./gna -- exp -L` will now print the list of available experiments:

.. code-block:: text

   Loading config file: config/gna/gnacfg.py
   Loading config file: config_local/gna/gnacfg.py
   Search paths:  ./macro/tutorial/ui/experiments
   UI exp list of experiments:
       exampleexp

As one can see from the output, GNA has read the user configuration file. The path to search for the experiments points
to the tutorial and there is only one experiment defined: `exampleexp`. This corresponds to the file
`./macro/tutorial/ui/experiments/exampleexp.py`.

We



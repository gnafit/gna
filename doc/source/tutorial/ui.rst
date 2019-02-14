Command line UI
^^^^^^^^^^^^^^^

GNA provides a set of tools accessible from the command line. Multiple tools, or modules, may be used within a single
GNA execution.
The functionality includes:

#. Manual parameters initialization.
#. Experimental data and models initialization.
#. Combined dataset definition.
#. Minimization.
#. Plotting.
#. And others.

Command line syntax
"""""""""""""""""""

GNA executable `./gna` is a wrapper, that expects a list of modules separated with `--`:

.. code-block:: bash

    ./gna -- module1 -- module2 [arg1] [arg2] -- module3 [arg1] -- ...

The arguments between two `--` strings are processed within a module itself.

A list of modules may be retrieved by `--list` or `--list-long` option:

.. code-block:: bash

    ./gna --list       # prints a list of modules
    ./gna --list-long  # prints a list of modules with docstrings (if available)

Default GNA modules are located in independent files in `./pylib/gna/ui/`

.. code-block:: text

    Listing available modules. Module search paths: ./pylib/gna/ui
        analysis             from ./pylib/gna/ui/analysis.pyc
        chi2                 from ./pylib/gna/ui/chi2.pyc
        contour              from ./pylib/gna/ui/contour.pyc
        covariance           from ./pylib/gna/ui/covariance.pyc
        ...

The module search path may be modified by extending `pkgpaths` option of `gnacfg.py`. It is done in a similar way to the
`bundlepaths` from :ref:`bundles <bundles_configuration>` tutorial.

Each GNA module when executed receives environment (`gna.env.env`) instance which is used as a common namespace for
parameters, evaluables, observables and others.

Each module defines its own arguments parser with `--help` command defined. For example the command

.. code-block:: bash

    ./gna -- plot-spectrum --help

will print the help for the arguments of the module `plot-spectrum`. The module itself may be found in
:download:`pylib/gna/ui/plot-spectrum.py <../../../pylib/gna/ui/plot-spectrum.py>`.

.. code-block:: text

    usage: gna plot-spectrum [-h] [-dp DIFFERENCE_PLOT] [-p DATA]
                             [--plot-type PLOT_TYPE] [--ratio RATIO RATIO] [--scale]
                             [-l Legend] [--plot-kwargs PLOT_KWARGS] [--drawgrid] [-s]
                             [--savefig SAVEFIG] [--title TITLE [TITLE ...]]
                             [--new-figure] [--xlabel XLABEL [XLABEL ...]]
                             [--ylabel YLABEL [YLABEL ...]]

    optional arguments:
      -h, --help            show this help message and exit
      -dp DIFFERENCE_PLOT, --difference-plot DIFFERENCE_PLOT
                            Subtract two obs, they MUST have the same binning
      -p DATA, --plot DATA

    ...

.. note::

   Similarly to argparse GNA is using `-/_` convention. In case symbol `-` is find in the module name, it is replaced
   with `_`. For example `./gna -- plot-spectrum` will search for `plot_spectrum` module, not `plot-spectrum`.

Example: initializing and plotting a model
""""""""""""""""""""""""""""""""""""""""""

Let us now use `gaussianpeak` module to create a simple module with integrated Gaussian function. The command

.. code-block:: bash

    python ./gna -- \
                 -- gaussianpeak --name peak

Creates a computational chain with integration and registers the output in the environment as `peak/spectrum`. This is
registered in the output as:

.. code-block:: text

    Add observable: peak/spectrum

This output now may be used from the other modules. For example, we may plot it with `plot-spectrum` module:

We will use multi-line commands for better readability. Note, that `\\` at the end of each line should have no spaces afterwards.

  .. ``

.. code-block:: bash

    python ./gna -- \
                 -- gaussianpeak --name peak \
                 -- plot-spectrum -p peak/spectrum  -s

The module `plot-spectrum` adds the output to the figure after `-p <name>` option. Argument `-s` enables `plot-spectrum`
to show the window with plotted figure after execution.

Both modules have options, that enable us to control the parameters. Let us define the energy range and number of bins
(see `./gna -- gaussianpeak --help` for reference). Also let us define the figure title and axes labels
(see `./gna -- plot-spectrum --help` for reference):

.. code-block:: bash

    python ./gna -- \
                 -- gaussianpeak --name peak --Emin 0 --Emax 5 --nbins 200 \
                 -- plot-spectrum -p peak/spectrum -t 'Gaussian Peak' -l 'Peak 1' --xlabel 'Energy, MeV' --ylabel '$dN/dE$' -s

Finally, let us save the image using `-o <filename.pdf>` option. With use `--latex` command to enable `matplotlib` use
latex for better rendering.

Also let us save the graph of the example model. The `graphviz` module reads the output name as the first argument.

.. code-block:: bash

  python ./gna -- \
               -- gaussianpeak --name peak --Emin 0 --Emax 5 --nbins 200 \
               -- graphviz peak/spectrum -o output/gna_ui_graph.pdf \
               -- plot_spectrum -p peak/spectrum -t 'Gaussian Peak' -l 'Peak 1' --xlabel 'Energy, MeV' --ylabel '$dN/dE$' -s --latex -o output/gna_ui_figure.pdf

The model is represented by the following graph:

.. figure:: ../../img/tutorial/ui/01_gna_ui_graph.png
   :align: center

   The computational chain created by the `gaussianpeak` module.

.. figure:: ../../img/tutorial/ui/01_gna_ui_figure.png
    :align: center

    The spectrum, created by the `gaussianpeak` module.

.. note::

    Do not forget to create `output` folder.



Fitting (TBD)
"""""""""""""


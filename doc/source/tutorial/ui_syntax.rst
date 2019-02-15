Command line syntax
"""""""""""""""""""

GNA executable `./gna` is a wrapper, that expects a list of modules separated with :code:`--`:

.. code-block:: bash

    ./gna -- module1 -- module2 [arg1] [arg2] -- module3 [arg1] -- ...

The arguments between two :code:`--` strings are processed within a module itself.

A list of modules may be retrieved by :code:`--list` or :code:`--list-long` option:

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

Each module defines its own arguments parser with :code:`--help` command defined. For example the command

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



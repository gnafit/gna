mpl_v1
""""""

The module implements most of the interactions with matplotlib, excluding the plotting itself. It changes global parameters of the matplotlib, decorates figures and saves images.
When `mpl-v1` is used to produce the output files the CWD from `env-cwd` is respected.


**Options**

*General*

    * ``-f, --figure`` -- create a new figure

    * ``-i, --interactive`` -- run mpl in interactive mode (*if not in batch mode*)

    * ``-b, --batch`` -- run mpl in batch mode (*if not in interactive mode*)

    * ``-l, --latex`` -- enable latex mode

    * ``-r, --rcparam, --rc`` -- load YAML dictionary with RC configuration

    * ``--style`` -- load matplotlib style

    * ``-v, --verbose`` -- define verbosity level 

*Axis modification*

    * ``-t, --title`` -- define the title of the figure

    * ``--xlabel, --xl`` -- define label of the x-axis

    * ``--ylabel, --yl`` -- define label of the y-axis

    * ``--xlim`` -- define the limits of the x-axis

        + positional arguments: *LOWER* *UPPER*


    * ``--ylim`` -- define the limits of the y-axis

        + positional arguments: *LOWER* *UPPER*


    * ``--legend`` -- define YAML dictionary to print as legend

    * ``--scale`` -- define scale of axis

        + positional arguments: *AXIS* *SCALE*


    * ``--powerlimits, --pl`` -- define powerlimits of axis

        + positional arguments: *AXIS* *PMIN* *PMAX*


    * ``--step-color`` -- step color of lines in the figure

    * ``-g, --grid`` --  draw grid

    * ``--minor-ticks, --mt`` -- enable minor ticks

    * ``--ticks-extra, --te`` -- add extra ticks

        + positional arguments: *AXIS* *TICK1* *TICK2* ...


*After plotting*

    * ``--tight-layout, --tl`` -- enable tight-layout mpl option

    * ``-o, --output`` -- define path to save the figure

    * ``-c, --close`` -- close the figure

    * ``-s, --show`` -- show the figure


**Examples**

    * Add labels and the title:

        .. code-block:: bash

            ./gna \
                -- mpl-v1 --xlabel 'Energy, MeV' --ylabel Entries -t 'The distribution'


    * Save a figure to the 'output.pdf' and then show it:

        .. code-block:: bash

            ./gna \
                -- mpl-v1 -o output.pdf -s


    * Create a new figure:

        .. code-block:: bash

            ./gna \
                -- mpl-v1 -f


    * Create a new figure of a specific size:

        .. code-block:: bash

            ./gna \
                -- mpl-v1 -f '{figsize: [14, 4]}'


    * Enable latex rendering:

        .. code-block:: bash

            ./gna \
                -- mpl-v1 -l


    * The module enables the user to tweak RC parameters by providing YAML dictionaries with options.
      Tweak matplotlib RC parameters to make all the lines of double width and setup power limits for the tick formatter:

        .. code-block:: bash

            ./gna \
                -- mpl-v1 -r 'lines.linewidth: 2.0' 'axes.formatter.limits: [-2, 2]'


    * An example of plotting, that uses the above mentioned options:

        .. code-block:: bash

            ./gna \
                -- env-cwd output/test-cwd \
                -- gaussianpeak --name peak_MC --nbins 50 \
                -- gaussianpeak --name peak_f  --nbins 50 \
                -- ns --name peak_MC --print \
                    --set E0             values=2    fixed \
                    --set Width          values=0.5  fixed \
                    --set Mu             values=2000 fixed \
                    --set BackgroundRate values=1000 fixed \
                -- ns --name peak_f --print \
                    --set E0             values=2.5  relsigma=0.2 \
                    --set Width          values=0.3  relsigma=0.2 \
                    --set Mu             values=1500 relsigma=0.25 \
                    --set BackgroundRate values=1100 relsigma=0.25 \
                -- plot-spectrum-v1 -p peak_MC.spectrum -l 'Monte-Carlo' --plot-type errorbar \
                -- plot-spectrum-v1 -p peak_f.spectrum -l 'Model (initial)' --plot-type hist \
                -- mpl-v1 --xlabel 'Energy, MeV' --ylabel entries -t 'Example plot' --grid \
                -- mpl-v1 -o figure.pdf -s
                

See also: *plot-spectrum-v1*, *plot-heatmap-v1*, *env-cwd*.


env_data_root
"""""""""""""

The module recursively copies all the outputs from the source location to the target location.
The outputs are converted to the ROOT objects. The produced data may then be saved with *save-root* module.

The overall idea is similar to the *env-data* module. Only TH1D, TH2D, TGraph are supported.
While histograms are written automatically for writing graphs the user need to use `-g` argument.


**Options**

    * ``-s, --root-source`` -- define root namespace to copy from

    * ``-t, --root-target`` -- define root namespace to copy to

    * ``-c, --copy`` -- define data to read and address to write

        + positional arguments: *FROM* *TO*


    * ``-g, --copy-graph`` -- define data to read (x,y) and address to write

        + positional arguments: *x* *y* *ADDRESS*


    * ``-v, --verbose`` -- define verbosity level

**Examples**

    * Write the data from all the outputs from the 'spectra' to 'output':

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak --nbins 50 \
                -- env-data-root -c spectra.peak output -vv \
                -- env-print -l 40

      The last command prints the data to stdout. The value width is limited to 40 symbols.

    * A common root for source and target paths may be set independently via `-s` and `-t` arguments.

      Store a graph read from 'fcn.x' and 'fcn.y' as 'output.fcn_graph':

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak --nbins 50 \
                -- env-data-root -s spectra.peak -g fcn.x fcn.y output.fcn_graph \
                -- env-print -l 40


See also: *env-data*, *save-yaml*, *save-pickle*, *save-root*.
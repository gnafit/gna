env_data
""""""""

The module recursively copies all the outputs from the source location to the target location. 
The outputs are converted to the dictionaries. Arrays, shapes, bin edges and object type are saved.

The produced data may then be saved with *save-yaml* and *save-pickle* modules.


**Options**

    * ``-s, --root-source`` -- define root namespace to copy from

    * ``-t, --root-target`` -- define root namespace to copy to

    * ``-C, --copy-as-is`` -- define data to read and address to write, no conversion

        + positional arguments: *FROM* *TO*


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
                -- env-data -c spectra.peak output -vv \
                -- env-print -l 40

      The last command prints the data to stdout. The value width is limited to 40 symbols.

    * A common root for source and target paths may be set independently via `-s` and `-t` arguments.
      There is also a special argument `-g` to combine graphs by reading X and Y arrays from different outputs.

      Store a graph read from 'fcn.x' and 'fcn.y' as 'output.fcn_graph':

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak --nbins 50 \
                -- env-data -s spectra.peak -g fcn.x fcn.y output.fcn_graph \
                -- env-print -l 40

    * Extra information may be saved with data. It should be provided as one ore more YAML dictionaries of the `-c` and `-g` arguments.
      The dictionaries will be used to update the target paths.

      Provide extra information:

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak --nbins 50 \
                -- env-data -c spectra.peak output '{note: extra information}' -vv \
                -- env-print -l 40


See also: *env-data-root*, *save-yaml*, *save-pickle*, *save-root*.
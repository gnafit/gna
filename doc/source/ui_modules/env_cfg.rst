env_cfg
"""""""

Global environment configuration UI. Enables verbosity for the debugging purposes.
All assignments and changes of the environment will be printed to stdout.


**Options**

    * ``-v, --verbose`` -- make environment verbose on set

    * ``-x, --exclude`` -- define keys to exclude

    * ``-i, --include`` -- define keys to include (only)


**Examples**

    * Enable verbosity:

        .. code-block:: bash

            ./gna \
                -- env-cfg -v \
                -- gaussianpeak --name peak_MC --nbins 50

The output may be filtered with `-x` and `-i` keys. Both support multiple arguments.

    * The `-x` option excludes matching keys:

        .. code-block:: bash

            ./gna \
                -- env-cfg -v -x fcn \
                -- gaussianpeak --name peak_MC --nbins 50

    * The `-i` option includes matching keys exclusively:

        .. code-block:: bash

            ./gna \
                -- env-cfg -v -i spectrum \
                -- gaussianpeak --name peak_MC --nbins 50


See also: *env-print*, *env-set*.
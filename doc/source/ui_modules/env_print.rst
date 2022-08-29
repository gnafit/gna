env_print
"""""""""""""

The module prints a given subtree within the env. The arguments are paths within env to be printed.
Paths may contains '.' which will be interpreted as a separator. It recursively prints key, type of the value and the value.

**Positional arguments**

    * ``paths`` -- define paths to print

**Options**

    * ``-l, --valuelen`` -- define value length

    * ``-k, --keylen`` -- define key length


**Examples**

Print the contents of the subtree 'spectra' and limit the value length to 40 symbols:

    .. code-block:: bash

        ./gna \
            -- gaussianpeak --name peak_MC --nbins 50 \
            -- env-print spectra -l 40

See also: *env-cfg*, *env-set*.
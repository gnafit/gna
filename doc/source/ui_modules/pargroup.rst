pargroup
""""""""

The module recursively selects parameters based on their status (free, constrained, fixed) and inclusion/exclusion mask. The list is stored in `env.future` and may be used by minimizers.

By default the module selects all the not fixed parameters: free and constrained.


**Positional arguments**

    * ``name`` -- defines parameter group name 

    * ``pars`` -- defines the parameters to store 

**Options**

    * ``--ns`` -- defines the namespace

    * ``-m, --modes`` -- take only parameters with specified property 

        + choose one or more from: *free*, *constrained*, *fixed*
        + default: *free* *constrained*


    * ``-x, --exclude`` -- define parameters to exclude

    * ``-i, --include`` -- define parameters to include exclusively

    * ``-a, --affect`` -- select only parameters that affect the output

    * ``-v, --verbose`` -- define verbosity level 

**Examples**

    * Select not fixed parameters from the namespace 'peak' and store as 'minpars':

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak \
                -- ns --name peak --print \
                    --set E0             values=2.5  free \
                    --set Width          values=0.3  relsigma=0.2 \
                    --set Mu             values=1500 relsigma=0.25 \
                    --set BackgroundRate values=1100 fixed \
                -- pargroup minpars peak -vv


    * Select only fixed parameters from the namespace 'peak' and store as 'minpars':

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak \
                -- ns --name peak --print \
                    --set E0             values=2.5  free \
                    --set Width          values=0.3  relsigma=0.2 \
                    --set Mu             values=1500 relsigma=0.25 \
                    --set BackgroundRate values=1100 fixed \
                -- pargroup minpars peak -m fixed -vv

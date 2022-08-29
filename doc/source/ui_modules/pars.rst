pars
""""

The module manages all parameters.


**Positional arguments**

    * ``root`` -- define the root namespace to work with


**Options**

    * ``-g, --group`` -- define parameter groups to work with

    * ``--free`` -- set parameter free

    * ``--variable`` -- set parameter as not fixed (*not to use with 'fix'/'fixed' option*)

    * ``--fix, --fixed`` -- set parameter as fixed (*not to use with 'variable' option*)

    * ``--randomize-constrained`` -- randomize normally distributed constrained parameters

    * ``--randomize-free`` -- define width to randomize normally distributed free parameters accroding to

    * ``--pop`` -- restore the previous value (*not to use with 'push' option*)

    * ``--push`` -- push the value and backup the previous one (*not to use with 'pop' option*)

    * ``--sigma`` -- set sigma value (*not to use with 'relsigma' option*)

    * ``--relsigma`` -- set relative sigma value (*not to use with 'sigma' option*)

    * ``-v, --verbose`` -- define verbosity level



**Example**

Set the parameters in the namespace 'peak' free:

    .. code-block:: bash
    
        ./gna \
            -- gaussianpeak --name peak \
            -- ns --name peak --print \
                --set E0             values=2.5  free \
                --set Width          values=0.3  relsigma=0.2 \
                --set Mu             values=1500 fixed \
                --set BackgroundRate values=1100 relsigma=0.25 \
            -- pars peak --free --variable -vv \
            -- ns --name peak --print



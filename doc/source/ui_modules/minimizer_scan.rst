minimizer_scan
""""""""""""""

The module initializes a hybrid minimizer which does a raster scan over a part of the variables.
The hybrid minimizer minimizes a set of parameters simply by scanning them, all the other parameters are minimized via regular minimizer at each point.
After the best fit is found, the minimizer performs a minimization over all the parameters.
The structure is similar with the `minimizer-v1`.

The module creates a minimizer instance which then may be used for a fit with `fit-v1` module or elsewhere.

It is important to note, that the grid parameters should also be included in the pargroup to be minimized.

The minimizer is stored in `env.future['minimizer']` under its name.


**Positional arguments**

    * ``name`` -- define name of the minimizer to use

    * ``statistic`` -- define the statistics function that has to be minimized (*created by stats module*)

    * ``pargroup`` -- define parameter groups for the minimization (*created by pargroups module*)

    * ``pargrid`` -- define name of the parameter grid to scan (*created by pargrid module*)

**Options**

    * ``-t, --type`` -- define type of minimizer 

        + choices: *minuit*, *minuit2*, *iminuit*
        + default: *minuit2*


    * ``-s, --strict`` -- raise an exception if a parameter is skipped because it does not affect the output

    * ``-v, --verbose`` -- define verbosity level

**Examples**

Create a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars', but do a raster scan over E0 (linear) and Width (log):

    .. code-block:: bash

        ./gna \
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
            -- dataset-v1 peak --theory-data peak_f.spectrum peak_MC.spectrum \
            -- analysis-v1 analysis --datasets peak \
            -- stats stats --chi2 analysis \
            -- pargroup minpars peak_f -vv \
            -- pargrid  scangrid --linspace  peak_f.E0    0.5 4.5 10 \
                                 --geomspace peak_f.Width 0.3 0.6 5 -v \
            -- minimizer-scan min stats minpars scangrid -vv \
            -- fit-v1 min -p --push \
            -- env-print fitresult.min

The *env-print* will print the status of the minimization, performed by the *fit-v1*. The intermediate results are saved in 'fitresults'.

See also: *minimizer-v1*, *fit-v1*, *stats*, *pargroup*.




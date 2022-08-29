minimizer_v1
""""""""""""

The module creates a minimizer instance which then may be used for a fit with *fit_v1* module or elsewhere.

The minimizer is stored in `env.future['minimizer']` under its name.


**Positional arguments**

    * ``name`` -- define name of the minimizer

    * ``statistic`` -- define the statistics function that has to be minimized (*created by stats module*)

    * ``pargroups`` -- define parameter groups for the minimization (*created by pargroups module*)

**Options**

    * ``-t, --type`` -- define type of minimizer

        + choices: *minuit*, *minuit2*, *iminuit*
        + default: *minuit2*


    * ``--minopts`` -- options that are passed to the minimizer

    * ``-s, --strict`` -- raise an exception if a parameter is skipped because it does not affect the output

    * ``--initial-value`` -- define what initial value to use

        + choices: *central* or *value*
        + default: *central*


    * ``-v, --verbose`` -- define verbosity level 


**Examples**

    * Create a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars':

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
                -- dataset-v1  --name peak --theory-data peak_f.spectrum peak_MC.spectrum \
                -- analysis-v1 --name analysis --datasets peak \
                -- stats stats --chi2 analysis \
                -- pargroup minpars peak_f -vv \
                -- minimizer-v1 min stats minpars -vv 


    * Create a minimizer and do a fit of a function 'stats' and a group of parameters 'minpars' using a `iminuit` minimizer:

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
                -- dataset-v1  --name peak --theory-data peak_f.spectrum peak_MC.spectrum \
                -- analysis-v1 --name analysis --datasets peak \
                -- stats stats --chi2 analysis \
                -- pargroup minpars peak_f -vv \
                -- minimizer-v1 min stats minpars -vv -t iminuit



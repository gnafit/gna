fit_v1
""""""

The module initializes a fit process with a minimizer, provided by *minimizer-v1*, *minimizer-scan* or others.

The fit result is saved to the `env.future['fitresult']` as a dictionary.


**Positional arguments**

    * ``minimizer`` -- define name of the minimizer to use

**Options**

    * ``-s, --set`` -- set best fit parameters 

    * ``-p, --push`` -- set (push) best fit parameters

    * ``-l, --label`` -- define the label to use to write results

    * ``--profile-errors`` -- calculate errors based on statistics profile

    * ``--scan`` -- calculate profiles for parameters

    * ``--covariance, --cov`` -- estimate covariance matrix

    * ``--simulate`` -- do nothing

    * ``--ndf`` -- read NDF for given chi2 from env

    * ``-v, --verbose`` -- print fit result to stdout

**Examples**

Perform a fit using a minimizer 'min':

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
            -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum \
            -- analysis-v1 analysis --datasets peak \
            -- stats stats --chi2 analysis \
            -- pargroup minpars peak_f -vv \
            -- minimizer-v1 min stats minpars -vv \
            -- fit-v1 min 

By default the parameters are set to initial after the minimization is done. It is possible to set the best fit parameters with option `-s` or with option `-p`. The latter option pushed the current values to the stack so they can be recovered in the future.

The result of the fit may be saved with *save_pickle* or *save_yaml* module.




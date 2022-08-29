analysis_v1
"""""""""""

Creates a named analysis, i.e. a triplet of theory, data and covariance matrix. The covariance matrix may be diagonal and contain only statistical uncertainties or contain a systematic part as well.


**Options**

    * ``-n, --name`` -- defines the name of the analysis (**required**)

    * ``-d, --datasets`` -- defines the list of datasets to use for the analysis (**required**)

    * ``-p, --cov-parameters`` -- defines the parameters for the covariance matrix

    * ``--cov-strict`` -- raises an exception if a parameter is skipped because it does not affect the output

    * ``-o, --observable`` -- defines the observable (model) to be fitted

    * ``--toymc`` -- use random sampling to variate the data/theory

        + choices: *covariance*, *poisson*, *normal*, *normalStats*, *asimov*


    * ``--toymc-source`` -- defines the source for '--toymc' option
 
        + choices: *data* or *theory*
        + default: *theory*


    * ``--covariance-updater`` -- defines name of the hook that triggers a covariance matrix update

    * ``-v, --verbose`` -- define verbosity level 


**Examples**

    * Initialize an analysis 'analysis' with a dataset 'peak':

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
                -- dataset-v1  --name peak --theory-data peak_f.spectrum peak_MC.spectrum -v \
                -- analysis-v1 --name analysis --datasets peak -v


    * Initialize an analysis 'analysis' with a dataset 'peak' and covariance matrix based on constrained parameters:

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
                -- dataset-v1 peak --theory-data peak_f.spectrum peak_MC.spectrum -v \
                -- pargroup covpars peak_f -m constrained \
                -- analysis-v1  analysis --datasets peak -p covpars -v


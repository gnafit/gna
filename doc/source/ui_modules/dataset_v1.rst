dataset-v1
""""""""""

The dataset-v1 module defines:

    * A pair of theory and data:

        + Observable (model) to be used as fitted function
        + Observable (data) to be fitted to

    * Statistical uncertainties (Pearson/Neyman) [theory/observation]
    * Or nuisance parameters

The dataset is added to the `env.future['spectra']`.

**Positional arguments**

    * ``name`` -- define the name of the dataset

**Options**

    * ``--pull`` -- parameters to be added as pull terms

    * ``--pull-groups`` -- parameter groups to be added as pull terms

    * ``--td, --theory-data`` -- defines the theory model and data inputs

        + positional arguments: *THEORY* *DATA*
    

    * --tdv, --theory-data-variance`` -- defines the theory model, data inputs and variance of the model 

        + positional arguments: *THEORY* *DATA* *VARIANCE*
    

    * --error-type -- defines the type of statistical error to be used with *--td* option 

        + choices: *pearson* or *neyman* 
        + default: *pearson*
    

    * -v, --verbose -- define verbosity level 


**Examples**

    * Initialize a dataset 'peak' with a pair of Theory/Data:

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak_MC --nbins 50 \
                -- gaussianpeak --name peak_f  --nbins 50 \
                -- ns --name peak_MC --print \
                   --set E0 values=2 fixed \
                   --set Width values=0.5 fixed \
                   --set Mu values=2000 fixed \
                   --set BackgroundRate values=1000 fixed \
                -- ns --name peak_f --print \
                   --set E0 values=2.5 relsigma=0.2 \
                   --set Width values=0.3 relsigma=0.2 \
                   --set Mu values=1500 relsigma=0.25 \
                   --set BackgroundRate values=1100 relsigma=0.25 \
                -- dataset-v1 --name peak --theory-data peak_f.spectrum peak_MC.spectrum -v
   

    * When a dataset is initialized from a nuisance terms it reads only constrained parameters from the namespace. Initialize a dataset 'nuisance' with a constrained parameters of 'peak\_f':

        .. code-block:: bash

            ./gna \
                -- gaussianpeak --name peak_MC --nbins 50 \
                -- gaussianpeak --name peak_f  --nbins 50 \
                -- ns --name peak_MC --print \
                   --set E0 values=2 fixed \
                   --set Width values=0.5 fixed \
                   --set Mu values=2000 fixed \
                   --set BackgroundRate values=1000 fixed \
                -- ns --name peak_f --print \
                   --set E0 values=2.5 relsigma=0.2 \
                   --set Width values=0.3 relsigma=0.2 \
                   --set Mu values=1500 relsigma=0.25 \
                   --set BackgroundRate values=1100 relsigma=0.25 \
                -- dataset-v1 --name nuisance --pull peak_f -v

Example for a sensitivity analysis with JUNO
""""""""""""""""""""""""""""""""""""""""""""

Combined JUNO+TAO sensitivity for a fit of IO to NO spectrum with full covariance.


**Full script**

.. code-block:: bash
 
    ./gna \
        -- env-cwd output/example_analysis -p nmo-sens-junotao-  \
        -- cmd-save command.sh  \
        -- exp --ns juno junotao_v06h -vv --binning-tao var-20 --lsnl-square  \
        -- comment 'Set initial mass ordering'  \
        -- nmo-set juno.pmns --set normal -vv  \
        -- pars juno.spectral_weights --free --variable  \
        -- snapshot juno.juno.final juno.asimov_juno  \
        -- snapshot juno.tao.final juno.asimov_tao  \
        -- pargroup oscpars juno.pmns -vv -m free constrained  \
        -- pargroup none -vv -m constrained  \
        -- pargroup --ns juno bkg acc_rate_norm lihe_rate_norm fastn_rate_norm alphan_rate_norm geonu_rate_norm atm_rate_norm reactors_elbl_rate_norm acc_rate_norm_tao lihe_rate_norm_tao -vv -m constrained  \
        -- pargroup --ns juno reac norm_reac thermal_power_scale energy_per_fission_scale fission_fractions_scale snf_norm offeq_scale -vv -m constrained  \
        -- pargroup --ns juno det norm_juno norm_tao eres_pars -vv -m constrained  \
        -- pargroup --ns juno lsnl lsnl_weight tao_escale -vv -m constrained  \
        -- pargroup --ns juno emodel_lsnl lsnl_weight -vv -m constrained  \
        -- pargroup --ns juno emodel_escale tao_escale -vv -m constrained  \
        -- pargroup --ns juno msw rho -vv -m constrained  \
        -- pargroup --ns juno specpars spectral_weights -v -m free  \
        -- dataset-v1 juno --theory-data-variance juno.juno.final juno.asimov_juno juno.juno.variance.all -vv  \
        -- dataset-v1 tao --theory-data-variance juno.tao.final juno.asimov_tao juno.tao.variance.all -vv  \
        -- dataset-v1 pmns --pull juno.pmns -vv  \
        -- analysis-v1 --name junotao --datasets juno tao --cov-parameters emodel_lsnl -vv  \
        -- analysis-v1 --name pmns --datasets pmns -vv  \
        -- dataset-v1 syst --pull-groups det emodel_escale reac bkg msw -vv  \
        -- analysis-v1 --name syst --datasets syst -vv  \
        -- stats chi2 --chi2 junotao --chi2 pmns --chi2 syst  \
        -- comment Set NMO: inverted  \
        -- nmo-set juno.pmns --set inverted -k ee -v  \
        -- minimizer-v1 junotao chi2 oscpars specpars det emodel_escale reac bkg msw -t iminuit --initial-value value -vv  \
        -- fit-v1 junotao --profile-errors juno.pmns.DeltaMSq13 juno.pmns.DeltaMSq12 juno.pmns.SinSqDouble12 juno.pmns.SinSqDouble13 --push -v  \
        -- env-set -r fitresult.junotao -y info '{model_variant: Dubna-subst, input: self, ordering: inverted, ordering-data: normal, combination: "JUNO+TAO", analysis: all, syst: all, covmat_juno: all, covmat_tao: all}'  \
        -- env-print fitresult -k 60 -l 40  \
        -- save-yaml fitresult -o fit.yaml -v  \
        -- save-pickle fitresult -o fit.pkl -v  \
        -- save-pickle fitresult fitresults -o plots.pkl -v  \
        -- mpl-v1 --figure -t 'JUNO fit' --xlabel 'Evis, MeV' --ylabel Entries/MeV  \
        -- plot-spectrum-v1 --plot-type hist --scale -p juno.asimov_juno -l NO data -p juno.juno.final -l 'Best fit: IO'  \
        -- mpl-v1 -o spectra.pdf  \
        -- env-cwd --print


**Step-by-step**

In the following every step of the sensitivity analysis is explained:

.. code-block:: bash
 
    ./gna \
        -- env-cwd output/example_analysis -p nmo-sens-junotao- \

This executes the GNA and sets the CWD to 'output/example_analysis'. If this directory doesn't exist, it is created.
All files that will be saved in the following process will get an additional prefix 'nmo-sens-junotao-'.


.. code-block:: bash
 
    -- cmd-save command.sh  \


This line will save the command line commands (i.e. the script that is executed) to the file 'nmo-sens-junotao-command.sh'.


.. code-block:: bash
 
    -- exp --ns juno junotao_v06h -vv --binning-tao var-20 --lsnl-square  \

This command loads the experiment definition from the file 'junotao_v06h.py' in the namespace 'juno' and passes the arguments 'binning-tao var-20' and 'lsnl-square' to the experiment definition.


.. code-block:: bash
 
    -- comment 'Set initial mass ordering'  \
    -- nmo-set juno.pmns --set normal -vv  \

The first line simply prints 'Set initial mass ordering' to stdout and the second line sets the NMO to normal ordering with the pmns parameters given in the namespace 'juno.pmns'.


.. code-block:: bash
 
    -- pars juno.spectral_weights --free --variable  \

Here the parameters in the namespace 'juno.spectral_weights' are set *free* and *not fixed*.


.. code-block:: bash
 
    -- snapshot juno.juno.final juno.asimov_juno  \
    -- snapshot juno.tao.final juno.asimov_tao  \

Now a snapshot of the outputs 'juno.juno.final' and 'juno.tao.final' is saved to 'juno.asimov_juno' and 'juno.asimov_tao' resp. which defines the experimental asimov data.


.. code-block:: bash
 
    -- pargroup oscpars juno.pmns -vv -m free constrained  \
    -- pargroup none -vv -m constrained  \
    -- pargroup --ns juno bkg acc_rate_norm lihe_rate_norm fastn_rate_norm alphan_rate_norm geonu_rate_norm atm_rate_norm reactors_elbl_rate_norm acc_rate_norm_tao lihe_rate_norm_tao -vv -m constrained  \
    -- pargroup --ns juno reac norm_reac thermal_power_scale energy_per_fission_scale fission_fractions_scale snf_norm offeq_scale -vv -m constrained  \
    -- pargroup --ns juno det norm_juno norm_tao eres_pars -vv -m constrained  \
    -- pargroup --ns juno lsnl lsnl_weight tao_escale -vv -m constrained  \
    -- pargroup --ns juno emodel_lsnl lsnl_weight -vv -m constrained  \
    -- pargroup --ns juno emodel_escale tao_escale -vv -m constrained  \
    -- pargroup --ns juno msw rho -vv -m constrained  \
    -- pargroup --ns juno specpars spectral_weights -v -m free  \

Several groups of parameters ('oscpars', 'none', 'bkg', 'reac', 'det', 'lsnl', emodel_lsnl', 'emodel_escale', 'msw', and 'specpars') in the namespace 'juno' are defined containing the parameters afterwards.
The parameters are also selected by their properties defined after ``-m`` option.


.. code-block:: bash
 
    -- dataset-v1 juno --theory-data-variance juno.juno.final juno.asimov_juno juno.juno.variance.all -vv  \

Now a dataset with name 'juno' is defined with theory-model 'juno.juno.final', data 'juno.asimov_juno', and variance 'juno.juno.variance.all'.


.. code-block:: bash
 
    -- dataset-v1 tao --theory-data-variance juno.tao.final juno.asimov_tao juno.tao.variance.all -vv  \

Another dataset with name 'tao' is defined with theory-model 'juno.tao.final', data 'juno.asimov_tao', and variance 'juno.tao.variance.all'.


.. code-block:: bash
 
    -- dataset-v1 pmns --pull juno.pmns -vv  \

A third dataset with name 'pmns' is defined by the pull-terms given by the parameters in the namespace 'juno.pmns'.


.. code-block:: bash
 
    -- analysis-v1 --name junotao --datasets juno tao --cov-parameters emodel_lsnl -vv  \

Now an analysis with name 'junotao' is created using the datasets 'juno' and 'tao' with covariance matrix created from the 'emodel_lsnl' parameter group.


.. code-block:: bash
 
    -- analysis-v1 --name pmns --datasets pmns -vv  \

A second analysis with name 'pmns' is created using the dataset 'pmns'.


.. code-block:: bash
 
    -- dataset-v1 syst --pull-groups det emodel_escale reac bkg msw -vv  \

A dataset with name 'syst' is defined by the parameter groups 'det', 'emodel_scale', 'reac', 'bkg', and 'msw' added as pull-terms.


.. code-block:: bash
 
    -- analysis-v1 --name syst --datasets syst -vv  \

Using the previously defined dataset 'syst' an analysis with name 'syst' is created.


.. code-block:: bash
 
    -- stats chi2 --chi2 junotao --chi2 pmns --chi2 syst  \

Next, the statistic with name 'chi2' is defined as a sum of three :math:`\chi^2` contributions given by the analyses 'junotao', 'pmns', and 'syst'. 


.. code-block:: bash

    -- comment Set NMO: inverted  \
    -- nmo-set juno.pmns --set inverted -k ee -v  \

As we want to fit a model with IO to data with NO, we now change the NMO. The first line again just prints 'Set NMO: inverted' to stdout, while the second line sets the NMO to inverted while keeping the mass splitting :math:`\Delta m_{ee}^2`.
The pmns parameters are again given in the namespace 'juno.pmns'.


.. code-block:: bash

    -- minimizer-v1 junotao chi2 oscpars specpars det emodel_escale reac bkg msw -t iminuit --initial-value value -vv  \

This command defines a minimizer with name 'junotao' of type 'iminuit' to minimize the statistic 'chi2'.
The minimization parameters are given by the parameter groups 'oscpars', 'specpars', 'det', 'emodel_escale', 'reac', 'bkg', and 'msw', while their initial value is set to their current value.


.. code-block:: bash

    -- fit-v1 junotao --profile-errors juno.pmns.DeltaMSq13 juno.pmns.DeltaMSq12 juno.pmns.SinSqDouble12 juno.pmns.SinSqDouble13 --push -v  \

The fit process is executed with the minimizer 'junotao' and the errors for the parameters 'juno.pmns.DeltaMSq13', 'juno.pmns.DeltaMSq12', 'juno.pmns.SinSqDouble12', and 'juno.pmns.SinSqDouble13' are calculated.
The best fit parameters are pushed to the fit model after the fit is finished.


.. code-block:: bash

    -- env-set -r fitresult.junotao -y info '{model_variant: Dubna-subst, input: self, ordering: inverted, ordering-data: normal, combination: "JUNO+TAO", analysis: all, syst: all, covmat_juno: all, covmat_tao: all}'  \

Additional information is added to the environment 'fitresult.junotao' as a new field 'info' containing the given dictionary.


.. code-block:: bash
 
    -- env-print fitresult -k 60 -l 40  \
    -- save-yaml fitresult -o fit.yaml -v  \
    -- save-pickle fitresult -o fit.pkl -v  \
    -- save-pickle fitresult fitresults -o plots.pkl -v  \

First, the environment 'fitresult' is printed to stdout with a maximum key-length of 60 symbols and a maximum value-length of 40 symbols.
Then, it is saved to the yaml and pickle files 'nmo-sens-junotao-fit.yaml' and 'nmo-sens-junotao-fit.pkl'.
Additionally, the environments 'fitresult' and 'fitresults' are saved to the pickle file 'nmo-sens-junotao-plots.pkl'. 


.. code-block:: bash

    -- mpl-v1 --figure -t 'JUNO fit' --xlabel 'Evis, MeV' --ylabel Entries/MeV  \
    -- plot-spectrum-v1 --plot-type hist --scale -p juno.asimov_juno -l NO data -p juno.juno.final -l 'Best fit: IO'  \
    -- mpl-v1 -o spectra.pdf  \

To plot the results, the matplotlib is configured to create a new figure with title 'JUNO fit' and labels 'Evis, MeV' for x-axis and 'Entries/MeV' for y-axis.
Then, the outputs 'juno.asimov_juno' with label 'NO data' and 'juno.juno.final' with label 'Best fit: IO' are plotted as a histogram that is scaled by the bin-width.
The third line saves the figure with the plot to the file 'nmo-sens-junotao-spectra.pdf'


.. code-block:: bash

    -- env-cwd --print
 
Last, this command prints a list of all processed files to stdout.
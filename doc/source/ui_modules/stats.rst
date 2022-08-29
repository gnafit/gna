stats
"""""

Build test statistic based on arbitrary sum of :math:`\chi^2` and logPoisson functions.


**Positional arguments**

    * ``name`` -- define name of the statistic 

**Options**

    * ``-c, --chi2`` -- defines analysis with :math:`\chi^2` contribution

    * ``--chi2-unbiased`` -- defines analysis with bias corrected :math:`\chi^2` contribution (:math:`+\log |V|`)

    * ``-p, --logpoisson`` -- defines analysis with logPoisson contribution

    * ``-P, --logpoisson-ratio`` -- defines analysis with log(Poisson ratio) contribution

    * ``--logpoisson-legacy`` -- defines analysis with logPoisson contribution (deprecated implementation)

    * ``--labels`` -- define list of node labels


**Examples**

Initialize an analysis 'analysis' with a dataset 'peak' and covariance matrix based on constrained parameters:

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
        -- analysis-v1  analysis --datasets peak -p covpars -v \
	-- stats chi2 --chi2 analysis

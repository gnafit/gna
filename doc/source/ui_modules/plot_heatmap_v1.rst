plot_heatmap_v1
"""""""""""""""

The module plots a 2-dimensional output as a heatmap.


**Positional arguments**

    * ``plot`` -- define list of observables to be plotted


**Options**

    * ``-l, --log`` -- use a log scale (*not to use with 'sym-log' option*)

    * ``--sym-log`` -- use a log scale and define linear interval around zero (*not to use with 'l'/'log' option*)

    * ``-f, --filter`` -- define list of filters to filter the matrix

        + choices: *triu*, *tril*, *diag*, *corr*, *llt*


    * ``--plot-kwargs`` -- all additional plotting options go here. They are applied for all plots


**Filters**

    * ``triu`` -- returns upper triangular of the matrix, lower is set to zero

    * ``tril`` -- returns lower triangular of the matrix, upper is set to zero

    * ``diag`` -- returns only diagonal of the matrix, all other values are set to zero

    * ``corr`` -- returns the correlation matrix

    * ``llt`` -- returns matrix multiplied by its transposed


**Example**

Plot a lower triangular matrix L — the Cholesky decomposition of the covariance matrix:

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
            -- pargroup minpars peak_f -vv -m free \
            -- pargroup covpars peak_f -vv -m constrained \
            -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum -vv \
            -- analysis-v1 peak --datasets peak -p covpars -v \
            -- env-print analysis \
            -- plot-heatmap-v1 analysis.peak.0.L -f tril \
            -- mpl-v1 --xlabel columns --ylabel rows -t 'Cholesky decomposition, L' -s

Here the filter 'tril' provided via `-f` ensures that only the lower triangular is plotted since it is not guaranteed that the upper matrix is reset to zero.



For more details on decorations and saving see *mpl-v1*.

See also: *mpl_v1*, *plot_spectrum_v1*.

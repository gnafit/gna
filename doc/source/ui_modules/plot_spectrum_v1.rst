plot_spectrum_v1
""""""""""""""""

The module plots 1 dimensional observables with matplotlib: plots, histograms and error bars.
The default way is to provide an observable after the `-p` option. The option may be used multiple times to plot multiple plots.


**Options**

*Data to plot*

    * ``-p, --plot`` -- list of observables to plot

    * ``-dp, --difference-plot, --diff`` -- plot the difference of two observables with equal binning

    * ``--lr, --log-ratio`` -- plot the log ratio of two observables

    * ``--ratio`` -- plot the ratio of two observables

*Additional options*

    * ``--vs`` -- define points over X axis to plot observable vs (*use only with plot-types 'plot' and 'ravelplot'*)

    * ``--plot-type`` -- define the type of the plot

        + choices: *bin_center*, *bar*, *hist*, *histo*, *errorbar*, *plot*, *ravelplot*


    * ``--scale`` -- scale histogram by bin width

    * ``--inverse`` -- inverse Y as 1/Y

    * ``--sqrt`` -- take sqrt from Y 

    * ``--index`` -- enable indexing for x-axis; DO NOT USE WITH --scale

    * ``--allow-diagonal, --diag`` -- use diagonal in case 2d array is passed

    * ``-l, --legend`` -- add legend to the plot, note that number of legends must match the number of plots

    * ``--plot-kwargs`` -- all additional plotting options go here. They are applied for all plots


**Example**

Plot two histograms, 'peak_MC' with error bars and 'peak_f' with lines:

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
            -- plot-spectrum-v1 -p peak_MC.spectrum -l 'Monte-Carlo' --plot-type errorbar \
            -- plot-spectrum-v1 -p peak_f.spectrum -l 'Model (initial)' --plot-type hist \
            -- mpl --xlabel 'Energy, MeV' --ylabel entries -t 'Example plot' --grid -s


For more details on decorations and saving see *mpl-v1*.

See also: *plot_heatmap_v1*.

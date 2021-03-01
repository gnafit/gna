#!/bin/bash

OUTPUT=output/tutorial_img/fit
mkdir -p $OUTPUT ^/dev/null
FIG1=$OUTPUT/01_fit_models_initial.png
FIG2=$OUTPUT/01_fit_models_fit.png
YAML=$OUTPUT/01_fit_models.yaml
GRAPH=$OUTPUT/01_fit_models_graph.png
OUT=$OUTPUT/01_fit_models.out

./gna \
      -- gaussianpeak --name peak_MC --nbins 50 \
      -- gaussianpeak --name peak_f  --nbins 50 \
      -- ns --name peak_MC --print \
            --set E0             values=2    fixed \
            --set Width          values=0.5  fixed \
            --set Mu             values=2000 fixed \
            --set BackgroundRate values=1000 fixed \
      -- ns --name peak_f --print \
            --set E0             values=2    free \
            --set Width          values=0.5  free \
            --set Mu             values=2000 free \
            --set BackgroundRate values=1000 free \
      -- dataset  --name peak --asimov-data peak_f/spectrum peak_MC/spectrum \
      -- analysis --name analysis --datasets peak \
      -- graphviz peak_f/spectrum -o $GRAPH --option rankdir TB \
      -- chi2 stats_chi2 analysis \
      -- ns --name peak_f --print \
            --set E0             value=3    \
            --set Width          value=0.2  \
            --set Mu             value=100  \
            --set BackgroundRate value=2000 \
      -- minimizer min minuit stats_chi2 peak_f \
      -- spectrum -p peak_MC/spectrum -l 'Monte-Carlo' --plot-type errorbar \
      -- spectrum  -p peak_f/spectrum -l 'Model (initial)' \
      -- mpl-v1 --xlabel 'Energy, MeV' --ylabel entries --grid -o $FIG1 \
      -- fit min -s -p -o $YAML \
      -- ns --print peak_f \
      -- mpl --figure \
      -- spectrum -p peak_MC/spectrum -l 'Monte-Carlo' --plot-type errorbar \
      -- spectrum -p peak_f/spectrum -l 'Best fit model' \
      -- mpl --xlabel 'Energy, MeV' --ylabel entries --grid \
                -o output/fit_01.pdf -o $FIG2 \
      > $OUT

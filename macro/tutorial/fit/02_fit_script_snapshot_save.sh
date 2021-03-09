#!/bin/bash

OUTPUT=output/tutorial_img/fit
mkdir -p $OUTPUT ^/dev/null
FIG1=$OUTPUT/02_fit_models_snapshot.png
YAML=$OUTPUT/02_fit_models_snapshot.yaml
GRAPH=$OUTPUT/02_fit_models_snapshot_graph.png
OUT=$OUTPUT/02_fit_models_snapshot.out

./gna \
      -- gaussianpeak --name peak  --nbins 50 \
      -- ns --name peak --print \
            --set E0             values=2    free \
            --set Width          values=0.5  free \
            --set Mu             values=2000 free \
            --set BackgroundRate values=1000 free \
      -- snapshot peak/spectrum peak/spectrum_MC --label 'Asimov MC' \
      -- dataset  --name peak --asimov-data peak/spectrum peak/spectrum_MC \
      -- analysis --name analysis --datasets peak \
      -- chi2 stats_chi2 analysis \
      -- minimizer min minuit stats_chi2 peak \
      -- ns --name peak --print \
            --set E0             value=3   \
            --set Width          value=0.2 \
            --set Mu             value=100  \
            --set BackgroundRate value=2000 \
      -- spectrum -p peak/spectrum_MC -l 'Asimov MC' --plot-type errorbar \
      -- spectrum -p peak/spectrum -l 'Model (initial)' \
      -- graphviz peak/spectrum -o $GRAPH --option rankdir TB \
      -- fit min -s -p -o $YAML \
      -- ns --print peak \
      -- spectrum  -p peak/spectrum -l 'Best fit' \
      -- mpl --xlabel 'Energy, MeV' --ylabel entries --grid \
             -o $FIG1 \
      > $OUT

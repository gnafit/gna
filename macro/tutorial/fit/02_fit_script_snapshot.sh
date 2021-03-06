#!/bin/bash

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
      -- graphviz peak/spectrum -o output/fit_02_graph.pdf \
      -- fit min -s -p -o output/fit_02.yaml \
      -- ns --print peak \
      -- spectrum -p peak/spectrum -l 'Best fit' \
      -- mpl --xlabel 'Energy, MeV' --ylabel entries --grid \
             -o output/fit_02.pdf -s

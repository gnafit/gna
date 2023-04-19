#!/usr/bin/env bash

set -euo pipefail

fcns="chi2 chi2-unbiased chi2-cnp-stat logpoisson logpoisson-ratio"

for fcn in $fcns
do
    ./gna \
          -- env-cwd output/test-stats-v1 \
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
          -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum --variable-error \
          -- analysis-v1 analysis --datasets peak \
          -- stats-v1 fcn --$fcn analysis --debug-min-steps \
          -- graphviz-v1 peak_f.spectrum -o stats_v1_graph_$fcn.pdf \
          -- ns --name peak_f --print \
                --set E0             value=3    \
                --set Width          value=0.2  \
                --set Mu             value=100  \
                --set BackgroundRate value=2000 \
          -- pargroup minpars peak_f -vv \
          -- minimizer-v2 min fcn minpars -vvv \
          -- plot-spectrum-v1 -p peak_MC.spectrum -l 'Monte-Carlo' --plot-type errorbar \
          -- plot-spectrum-v1 -p peak_f.spectrum -l 'Model (initial)' \
          -- mpl-v1 --xlabel 'Energy, MeV' --ylabel entries --grid \
          -- fit-v1 min -s -p --profile-errors peak_f.Mu \
          -- ns --print peak_f \
          -- spectrum -p peak_f/spectrum -l 'Best fit model' \
          -- env-print statistic.fcn \
          -- env-print fitresult.min \
          -- mpl -s

done

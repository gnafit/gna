#!/usr/bin/env bash

set -euo pipefail

fcns="chi2 chi2-unbiased chi2-cnp-stat logpoisson logpoisson-ratio"

for fcn in $fcns
do
    ./gna \
          -- env-cwd output/test-stats \
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
          -- graphviz peak_f/spectrum -o fit_01_graph.pdf \
          -- stats fcn --$fcn analysis \
          -- ns --name peak_f --print \
                --set E0             value=3    \
                --set Width          value=0.2  \
                --set Mu             value=100  \
                --set BackgroundRate value=2000 \
          -- pargroup minpars peak_f -vv \
          -- minimizer-v1 min fcn minpars -t iminuit \
          -- spectrum -p peak_MC/spectrum -l 'Monte-Carlo' --plot-type errorbar \
          -- spectrum -p peak_f/spectrum -l 'Model (initial)' \
          -- mpl-v1 --xlabel 'Energy, MeV' --ylabel entries --grid \
          -- fit-v1 min -s -p --profile-errors peak_f.Mu \
          -- ns --print peak_f \
          -- spectrum -p peak_f/spectrum -l 'Best fit model' \
          -- env-print fitresult.min \
          # -- mpl -s
done

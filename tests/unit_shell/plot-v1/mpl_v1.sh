#!/usr/bin/env bash

./gna \
      -- env-cwd output/test-mpl \
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
      -- mpl-v1 --xlabel 'Energy, MeV' --ylabel entries -t 'Example plot' --grid \
      -- mpl-v1 -o mpl.pdf

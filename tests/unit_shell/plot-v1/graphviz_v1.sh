#!/usr/bin/env bash

ipython3 --pdb -- ./gna \
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
    -- analysis-v1 analysis --datasets peak -p covpars -v \
    -- stats stats --chi2 analysis \
    -- graphviz peak_f.spectrum -o graphviz-parameters-example.pdf --ns

#!/usr/bin/env bash

ipython3 --pdb -- ./gna \
      -- env-cwd output/test-scan \
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
      -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum \
      -- analysis-v1 analysis --datasets peak \
      -- stats stats --chi2 analysis \
      -- pargroup pars peak_f -vv \
      -- pargrid scangrid --linspace peak_f.E0 0 10 100  \
                          --linspace peak_f.Width 0.1 2 100 \
      -- minimizer-scan min stats pars scangrid -vv \
      -- scan-v1 min --output test.hdf5

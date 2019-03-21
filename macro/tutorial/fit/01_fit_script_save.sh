#!/bin/bash

OUTPUT=output/tutorial
mkdir -p $OUTPUT ^/dev/null
FIG1=$OUTPUT/01_fit_models_initial.png
FIG2=$OUTPUT/01_fit_models_fit.png
YAML=$OUTPUT/01_fit_models.yaml

./gna \
      -- gaussianpeak --name peak_MC --nbins 50 \
      -- gaussianpeak --name peak_f  --nbins 50 \
      -- ns --name peak_MC --print \
            --set E0             values=2    fixed \
            --set Width          values=0.5  fixed \
            --set Mu             values=2000 fixed \
            --set BackgroundRate values=1000 fixed \
      -- ns --name peak_f --print \
            --set E0             values=4    relsigma=0.2 \
            --set Width          values=0.2  relsigma=0.2 \
            --set Mu             values=100  relsigma=0.25 \
            --set BackgroundRate values=2000 relsigma=0.25 \
      -- spectrum -p peak_MC/spectrum -l 'Monte-Carlo' --plot-type errorbar \
                  --xlabel 'Energy, MeV' large --ylabel entries large --grid \
      -- spectrum  -p peak_f/spectrum -l 'Model (initial)' -o $FIG1 \
      -- dataset  --name peak --asimov-data peak_f/spectrum peak_MC/spectrum \
      -- analysis --name analysis --datasets peak \
      -- chi2 stats analysis \
      -- minimizer min minuit stats peak_f \
      -- fit min -s -p -o $YAML \
      -- ns --print peak_f \
      -- spectrum --new-figure -p peak_MC/spectrum -l 'Monte-Carlo' --plot-type errorbar \
                  --xlabel 'Energy, MeV' large --ylabel entries large --grid \
      -- spectrum  -p peak_f/spectrum -l 'Best fit model' -o $FIG2

if test "$1" = "save"
then
    mkdir -p doc/img/tutorial/fit ^/dev/null && cp -v $FIG1 $FIG2 $YAML doc/img/tutorial/fit/
fi
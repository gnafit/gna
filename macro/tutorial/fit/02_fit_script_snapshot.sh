#!/bin/bash

OUTPUT=output/tutorial
mkdir -p $OUTPUT ^/dev/null
FIG1=$OUTPUT/02_fit_models_snapshot.png
YAML=$OUTPUT/02_fit_models_snapshot.yaml
GRAPH=$OUTPUT/02_fit_models_snapshot_graph.png

./gna \
      -- gaussianpeak --name peak  --nbins 50 \
      -- ns --name peak --print \
            --set E0             values=2    relsigma=0.2  \
            --set Width          values=0.5  relsigma=0.2  \
            --set Mu             values=2000 relsigma=0.25 \
            --set BackgroundRate values=1000 relsigma=0.25 \
      -- snapshot peak/spectrum peak/spectrum_MC --label 'Asimov MC' \
      -- dataset  --name peak --asimov-data peak/spectrum peak/spectrum_MC \
      -- analysis --name analysis --datasets peak \
      -- chi2 stats_chi2 analysis \
      -- minimizer min minuit stats_chi2 peak \
      -- ns --name peak --print \
            --set E0             value=4   \
            --set Width          value=0.2 \
            --set Mu             value=100  \
            --set BackgroundRate value=2000 \
      -- plot-spectrum -p peak/spectrum_MC -l 'Asimov MC' --plot-type errorbar \
                       --xlabel 'Energy, MeV' large --ylabel entries large --grid \
      -- graphviz peak/spectrum -o $GRAPH \
      -- plot-spectrum -p peak/spectrum -l 'Model (initial)' \
      -- fit min -s -p -o $YAML \
      -- ns --print peak \
      -- plot-spectrum  -p peak/spectrum -l 'Best fit' -o $FIG1

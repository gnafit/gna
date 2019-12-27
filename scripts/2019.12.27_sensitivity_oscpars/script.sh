#!/usr/bin/env bash

ipython --pdb -- \
  ./gna \
      -- exp  --ns juno juno_sensitivity_v01 -vv --energy-model lsnl eres --free osc --oscprob vacuum \
      -- snapshot juno/AD1 juno/asimov \
      -- dataset --name juno --asimov-data juno/AD1 juno/asimov \
      -- analysis --name juno --datasets juno \
      -- chi2 stats-chi2 juno \
      -- minimizer min minuit stats-chi2 juno.pmns \
      -- fit min -sp -o output/fit.yaml --profile juno.pmns.DeltaMSq23

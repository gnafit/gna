#!/usr/bin/env bash

output=output/2020.01.15_solar
mkdir -p $output

./gna \
  -- exp \
       --ns juno_vac juno_sensitivity_v01 -vv \
       --energy-model lsnl multieres \
       --subdetectors-number 5 \
       --multieres concat \
       --estep 0.01 \
       --eres-npe 1200 \
       --free osc \
       --oscprob vacuum \
       --dm ee \
       --spectrum initial \
       --reactors pessimistic nohz \
  -- exp \
       --ns juno_mat juno_sensitivity_v01 -vv \
       --energy-model lsnl multieres \
       --subdetectors-number 5 \
       --multieres concat \
       --estep 0.01 \
       --eres-npe 1200 \
       --free osc \
       --oscprob matter \
       --dm ee \
       --spectrum initial \
       --reactors pessimistic nohz \
  -- ns \
       --set juno_vac.pmns.SinSqDouble13 free=0 \
       --set juno_mat.pmns.SinSqDouble13 free=0 \
  -- ns \
       --output $output/parameters.yaml \
  -- dataset \
       --name juno \
       --asimov-data juno_vac/AD1 juno_mat/AD1 \
  -- analysis \
       --name juno \
       --datasets juno \
       --cov-parameters juno_vac.norm juno_vac.pmns.SinSqDouble13 juno_vac.thermal_power \
                        juno_vac.spectrum juno_vac.subdetector_fraction juno_vac.Npescint juno_vac.kC juno_vac.birks \
  -- chi2 stats-chi2 juno \
  -- graphviz juno_mat/AD1 -o $output/graph.dot \
  -- minimizer min minuit stats-chi2 juno_vac.pmns \
       --drop-constrained \
  -- fit min -sp -o $output/fit.yaml \
       --profile juno_vac.pmns.DeltaMSqEE juno_vac.pmns.DeltaMSq12 juno_vac.pmns.SinSqDouble12 -a label 'x2 bins' \
  | tee $output/cmd.out

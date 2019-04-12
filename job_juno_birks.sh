#!/bin/bash

output=output/2019.04.12_juno_nl_example
mkdir -p $output

python gna -- ns \
    -- nl_juno --name ju --with-birks \
    -- ns --name ju --print \
    -- spectrum --plot ju/AD1_noeffects -l no_effect_spectrum --plot ju/AD1_lsnl_bc -l add_lsnl --plot ju/AD1_Eres -l add_Eres --grid -o $output/juno_nl_figure.pdf -s \
    -- graphviz ju/AD1_Eres -o $output/juno_nl_graph.dot --namespace ju

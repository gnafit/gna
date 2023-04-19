#!/usr/bin/env bash

./gna \
    -- gaussianpeak --name peak --nbins 50 \
    -- fcn sqrt peak.spectrum peak.stat_unc -l 'Stat error' \
    -- ns --name peak --print --set Mu values=2000 \
    -- seed -s 1 \
    -- toymc peak.spectrum peak.mc_asimov      -t asimov      \
    -- toymc peak.spectrum peak.mc_poisson     -t poisson     \
    -- toymc peak.spectrum peak.mc_normalStats -t normalStats \
    -- toymc peak.spectrum peak.mc_normal      -t normal     -u peak.stat_unc \
    -- seed -s 2 \
    -- toymc peak.spectrum peak.mc_poisson2    -t poisson     \
    -- seed -s 1 \
    -- toymc peak.spectrum peak.mc_poisson1    -t poisson     \
    -- mpl-v1 -f -t 'Distributions' \
    -- plot-spectrum-v1 -p peak.mc_asimov      -l Asimov          --plot-type hist     \
    -- plot-spectrum-v1 -p peak.mc_poisson     -l Poisson         --plot-type errorbar --plot-kwargs '{color: red}' \
    -- plot-spectrum-v1 -p peak.mc_normalStats -l 'Normal: stats' --plot-type errorbar --plot-kwargs '{color: green}' \
    -- plot-spectrum-v1 -p peak.mc_normal      -l 'Normal'        --plot-type errorbar --plot-kwargs '{color: blue}' \
    -- mpl-v1 -f -t 'Ratio to input' \
    -- plot-spectrum-v1 --ratio peak.mc_asimov      peak.spectrum -l Asimov          --plot-type hist     \
    -- plot-spectrum-v1 --ratio peak.mc_poisson     peak.spectrum -l Poisson         --plot-type hist --plot-kwargs '{color: red}' \
    -- plot-spectrum-v1 --ratio peak.mc_normalStats peak.spectrum -l 'Normal: stats' --plot-type hist --plot-kwargs '{color: green}' \
    -- plot-spectrum-v1 --ratio peak.mc_normal      peak.spectrum -l 'Normal'        --plot-type hist --plot-kwargs '{color: blue}' \
    -- mpl-v1 -f -t 'Seeds: poisson' \
    -- plot-spectrum-v1 -p peak.mc_poisson  -l 'seed=1' --plot-type errorbar --plot-kwargs '{color: red, alpha: 0.5}' \
    -- plot-spectrum-v1 -p peak.mc_poisson2 -l 'seed=2' --plot-type errorbar --plot-kwargs '{color: green}' \
    -- plot-spectrum-v1 -p peak.mc_poisson1 -l 'seed=1' --plot-type errorbar --plot-kwargs '{color: blue, alpha: 0.5}' \
    -- mpl-v1 -f -t 'Seeds: poisson, ratio' \
    -- plot-spectrum-v1 --log-ratio peak.mc_poisson  peak.mc_poisson1 \
    # -- graphviz peak.spectrum -o output/gaussianpeak_mc.dot \
    # -- mpl-v1 -s

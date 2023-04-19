#!/usr/bin/env bash

./gna \
    -- env-cwd output/test-cwd \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data-root -c spectra output \
    -- env-data-root -s spectra.peak -g fcn.x fcn.y output.fcn_graph \
    -- env-print -l 40 \
    -- save-root output -o output.root

#!/usr/bin/env bash

./gna \
    -- gaussianpeak --name peak0 --nbins 50 \
    -- env-data-root -c spectra.peak0 output -vv \
    -- env-print -l 40 \
    -- gaussianpeak --name peak1 --nbins 50 \
    -- env-data-root -s spectra.peak1 -g fcn.x fcn.y output.fcn_graph \
    -- env-print -l 40

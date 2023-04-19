#!/usr/bin/env bash

./gna \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data -c spectra.peak output -vv \
    -- env-print -l 40 \
    -- env-data -s spectra.peak -g fcn.x fcn.y output.fcn_graph \
    -- env-print -l 40 \
    -- env-data -c spectra.peak output '{note: extra information}' -vv \
    -- env-print -l 40

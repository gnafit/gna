#!/usr/bin/env bash

./gna \
    -- env-cwd output/test-cwd \
    -- gaussianpeak --name peak --nbins 50 \
    -- env-data -c spectra output '{note: extra information}' -vv \
    -- env-print -l 40 \
    -- save-pickle output -o output.pkl

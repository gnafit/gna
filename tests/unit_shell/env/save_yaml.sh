#!/usr/bin/env bash

./gna \
    -- env-cwd output/test-cwd \
    -- gaussianpeak --name peak --nbins 5 \
    -- env-data -c spectra.peak output '{note: extra information}' -vv \
    -- env-print -l 40 \
    -- save-yaml output -o output.yaml

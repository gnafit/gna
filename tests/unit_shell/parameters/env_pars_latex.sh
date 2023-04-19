#!/usr/bin/env bash

./gna \
    -- env-cwd output/test-cwd \
    -- gaussianpeak --name peak \
    -- env-pars-latex peak -o output.tex

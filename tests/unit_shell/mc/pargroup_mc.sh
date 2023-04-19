#!/usr/bin/env bash

./gna \
    -- seed -s 1 -v \
    -- gaussianpeak --name peak --nbins 50 \
    -- plot-spectrum-v1 -p peak.spectrum -l Initial --plot-type hist \
    -- ns --print peak --print-long \
    -- pargroup mcpars1 peak.E0 peak.Width -vv \
    -- pargroup mcpars2 peak.Mu peak.BackgroundRate -vv \
    -- pars -g mcpars2 --correlation 0.9 \
    -- pargroup-mc mcpars1 mcpars2 -vv \
    -- ns --print peak \
    -- plot-spectrum-v1 -p peak.spectrum -l Modified --plot-type hist \
    # -- mpl-v1 -s

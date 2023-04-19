#!/usr/bin/env bash

./gna \
    -- gaussianpeak --name peak \
    -- ns --name peak --print \
          --set E0             values=2.5  free \
          --set Width          values=0.3  relsigma=0.2 \
          --set Mu             values=1500 relsigma=0.25 \
          --set BackgroundRate values=1100 fixed \
    -- pargroup minpars peak -m fixed -vv

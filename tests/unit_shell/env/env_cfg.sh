#!/usr/bin/env bash

./gna \
    -- env-cfg -v \
    -- gaussianpeak --name peak_MC0 --nbins 50 \
    -- env-cfg -v -x fcn \
    -- gaussianpeak --name peak_MC1 --nbins 50 \
    -- env-cfg -v -i spectrum \
    -- gaussianpeak --name peak_MC2 --nbins 50

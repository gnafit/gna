#!/usr/bin/env bash

./gna \
    -- comment Initialize a gaussian peak with default configuration and 50 bins \
    -- gaussianpeak --name peak_MC --nbins 50 \
    -- cmd-save command.sh

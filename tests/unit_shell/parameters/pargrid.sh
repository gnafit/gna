#!/usr/bin/env bash

./gna \
    -- gaussianpeak --name peak \
    -- pargrid scangrid --linspace peak.E0 0.5 4.5 10 -vv

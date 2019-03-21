#!/bin/bash

python gna -- ns \
    --define ju.Qp0 central=0.0065 sigma=0 \
    --define ju.Qp1 central=0.00015 sigma=0 \
    --define ju.Qp2 central=1341.38 sigma=0 \
    --define ju.Qp3 central=1.0 sigma=0 \
    -- nl_juno --name ju --with-mine \
    -- spectrum --plot ju/AD1_noeffects -l no_effect_spectrum --plot ju/AD1_mine -l add_lsnl --plot ju/AD1_Eres -l add_Eres

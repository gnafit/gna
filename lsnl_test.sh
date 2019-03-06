#!/bin/bash

# plot multi lsnl curve
./draw_lsnl.sh \


#choose the normalization point by scaling the lsnl curve

python gna lsnl_norm --name test -- spectrum --plot test/spectrum2 -l orginal --plot test/spectrum3 -l scale_by_2 \


# check the lsnl matrix, the 1st matrix is no nonlinearity, the 2nd matrix is to multiply all the energy points by 1.1

python gna drawlsnlmat --name j \

# check the lsnl in juno experiment
python gna nl_juno --name ju --with-mine -- spectrum --plot ju/AD1_noeffects -l  no_effect_spectrum --plot ju/AD1_mine -l add_lsnl --plot ju/AD1_Eres -l add_Eres \

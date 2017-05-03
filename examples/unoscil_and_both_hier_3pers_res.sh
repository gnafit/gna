#!/bin/bash
python gna juno --name juno_nh --binning AD1 1 10 500 --backgrounds geo \
    -- juno --name juno_ih --binning AD1 1 10 500 --backgrounds geo \
    -- ns --value juno_ih.oscillation.Alpha inverted \
    -- spectrum --plot juno_nh/AD1 -l NH --plot juno_ih/AD1 -l IH \
    --plot juno_nh/AD1_unoscillated -l Unoscillated  --plot_kwargs "{linewidth: 1.2}" 

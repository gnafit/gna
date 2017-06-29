#!/bin/bash
python gna juno --name juno_nh --binning AD1 1 10 500 --backgrounds geo -- juno --name juno_ih --binning AD1 1 10 500 --backgrounds geo \
    -- ns --value juno_ih.oscillation.Alpha inverted \
    -- spectrum --plot juno_nh/AD1 -l NH --plot juno_ih/AD1 -l IH \
    -dp juno_nh/AD1 -dp juno_ih/AD1 -l NH-IH  --plot-kwargs "{linewidth: 1.2}" 

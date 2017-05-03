#!/bin/bash
python gna ns --define common.Eres_b central=0.05 sigma=0.0 --push common \
    -- juno --name juno_nh --binning AD1 1 10 700 --backgrounds geo \
    -- juno --name juno_ih --binning AD1 1 10 700 --backgrounds geo \
    -- ns --value juno_ih.oscillation.Alpha inverted \
    -- spectrum -dp juno_nh/AD1 -dp juno_ih/AD1 -l NH-IH \
    --plot juno_nh/AD1 -l NH --plot juno_ih/AD1 -l IH  --plot_kwargs "{linewidth: 1.2}" 

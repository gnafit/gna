#!/bin/bash
python gna juno --name juno_nh --binning AD1 1 10 700 --backgrounds geo  \
    -- juno --name juno_ih --binning AD1 1 10 700 --backgrounds geo \
    -- ns --value juno_ih.oscillation.Alpha inverted \
    -- spectrum  --plot juno_nh/AD1_noeffects -l NH --plot juno_ih/AD1_noeffects -l IH \
    -dp juno_nh/AD1_noeffects -dp juno_ih/AD1_noeffects -l NH-IH  --plot_kwargs "{linewidth: 1.2}" 

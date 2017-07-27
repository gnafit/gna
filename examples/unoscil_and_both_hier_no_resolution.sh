#!/bin/bash
python gna juno --name juno_nh --binning AD1 1 10 1000 --backgrounds geo -- juno --name juno_ih --binning AD1 1 10 1000 --backgrounds geo \
    -- ns --value juno_ih.oscillation.Alpha inverted \
    -- spectrum --plot-type histo --plot juno_nh/AD1_noeffects -l NH --plot juno_ih/AD1_noeffects -l IH \
    --plot juno_nh/AD1_unoscillated_with_bkg -l "Unoscillated with geo"  --plot-kwargs "{linewidth: 1.2}" 

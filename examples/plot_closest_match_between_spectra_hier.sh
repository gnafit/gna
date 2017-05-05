#!/bin/bash
#Run the fit_the_normal_with_inverted_get_best_dMsqEE.sh to find constant
python gna juno --name juno_nh --backgrounds geo --binning AD1 1 10 500 \
        -- juno --name juno_ih --backgrounds geo --binning AD1 1 10 500 \
        -- ns --value juno_ih.oscillation.Alpha inverted \
              --value juno_ih.oscillation.DeltaMSqEE 0.00251722 \
        -- spectrum --plot juno_nh/AD1 -l NH --plot juno_ih/AD1 -l IH \
           -dp juno_nh/AD1 -dp juno_ih/AD1 -l NH-IH --plot_type histo \
           --plot_kwargs "{'linewidth': '1.2'}"


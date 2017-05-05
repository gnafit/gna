#!/bin/bash
python gna juno --name juno_nh --backgrounds geo --binning AD1 1 10 500 \
        -- juno --name juno_ih --backgrounds geo --binning AD1 1 10 500 \
        -- ns --value juno_ih.oscillation.Alpha inverted \
        -- dataset --name fit_hier_data --asimov-data juno_ih/AD1 juno_nh/AD1 \
        -- analysis --name fit_hier -d fit_hier_data -o juno_ih/AD1 \
        -- chi2 chi2 fit_hier \
        -- minimizer min minuit chi2 juno_ih.oscillation.DeltaMSqEE \
        -- fit min

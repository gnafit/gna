#!/bin/sh
python gna \
-- ns --define common.Eres_b central=0.03 sigma=0.0   --push common \
-- juno --name juno_ih --ibd first --binning AD1 1 10 1000 \
-- juno --name juno_nh --ibd first --binning AD1 1 10 1000 \
-- juno --name juno_fake_data --ibd first --binning AD1 1 10 1000 \
-- ns --value juno_ih.oscillation.Alpha inverted  \
      --value juno_fake_data.oscillation.Alpha inverted \
-- dataset --name asimov_ih --asimov-data juno_ih/AD1 juno_fake_data/AD1 \
-- dataset --name pull_dm_12_i --pull juno_ih.oscillation.DeltaMSq12 \
-- dataset --name pull_sin_12_i --pull juno_ih.oscillation.SinSq12 \
-- dataset --name pull_sin_13_i --pull juno_ih.oscillation.SinSq13 \
-- analysis --name inv_inv -d asimov_ih pull_dm_12_i pull_sin_13_i pull_sin_12_i \
   -o juno_ih/AD1 juno_ih.oscillation.SinSq12 \
      juno_ih.oscillation.SinSq13 juno_ih.oscillation.DeltaMSq12 \
-- chi2 chi2_inv inv_inv \
-- minimizer min_ih minuit chi2_inv juno_ih.oscillation.DeltaMSq12 \
   juno_ih.oscillation.SinSq13 juno_ih.oscillation.SinSq12 \
-- scan --lingrid juno_ih.oscillation.DeltaMSqEE  2.42e-3 2.58e-3 100 \
   --minimizer min_ih --verbose --output /tmp/out_ih.hdf5 \
 \
-- dataset --name asimov_nh --asimov-data juno_nh/AD1 juno_fake_data/AD1 \
-- dataset --name pull_dm_12_n --pull juno_nh.oscillation.DeltaMSq12 \
-- dataset --name pull_sin_12_n --pull juno_nh.oscillation.SinSq12 \
-- dataset --name pull_sin_13_n --pull juno_nh.oscillation.SinSq13 \
-- analysis --name nor_inv -d asimov_nh pull_dm_12_n pull_sin_13_n pull_sin_12_n \
   -o juno_nh/AD1 juno_nh.oscillation.SinSq12 \
      juno_nh.oscillation.SinSq13 juno_nh.oscillation.DeltaMSq12 \
-- chi2 chi2_nor nor_inv \
-- minimizer min_nh minuit chi2_nor juno_nh.oscillation.DeltaMSq12 \
   juno_nh.oscillation.SinSq13 juno_nh.oscillation.SinSq12 \
-- scan --lingrid juno_nh.oscillation.DeltaMSqEE  2.42e-3 2.58e-3 100 \
   --minimizer min_nh --verbose --output /tmp/out_nh.hdf5 \

# -- contour --no-shift --chi2 /tmp/out_ih.hdf5  --minimizer min_ih --plot chi2minplt \
   # --xlabel "\$\Delta m^2_{ee}\$" --ylabel "\$ \Delta \chi^2 \$"    --show


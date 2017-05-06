#!/bin/sh
python gna \
-- ns --define common.Eres_b central=0.03 sigma=0.0   --push common \
-- juno  --name juno_ih \
-- juno --name juno_nh \
-- juno --name juno_fake_data \
-- ns --value juno_ih.oscillation.Alpha inverted \
      --value juno_fake_data.oscillation.Alpha inverted \
-- dataset --name asimov_ih --asimov-data juno_nh/AD1 juno_fake_data/AD1 \
-- dataset --name pull_dm_12 --pull juno_nh.oscillation.DeltaMSq12 \
-- dataset --name pull_sin_12 --pull juno_nh.oscillation.SinSq12 \
-- dataset --name pull_sin_13 --pull juno_nh.oscillation.SinSq13 \
-- analysis --name testhier -d asimov_ih pull_dm_12 pull_sin_13 pull_sin_12 \
   -o juno_nh/AD1 juno_nh.oscillation.SinSq12 \
      juno_nh.oscillation.SinSq13 juno_nh.oscillation.DeltaMSq12 \
-- chi2 chi2 testhier \
-- minimizer min_nh minuit chi2 juno_nh.oscillation.DeltaMSq12 \
   juno_nh.oscillation.SinSq13 juno_nh.oscillation.SinSq12 \
-- scan --lingrid juno_nh.oscillation.DeltaMSqEE  2.42e-3 2.58e-3 100 \
   --minimizer min_nh --verbose --output /tmp/out_nh.hdf5 \
-- contour --chi2 /tmp/out_nh.hdf5  --minimizer min_nh --plot chi2minplt \
   --ylim -15 20 --xlabel "\$\Delta m^2_{ee}\$" --ylabel "\$ \Delta \chi^2 \$"    --show

# -- minimizer min minuit chi2 juno_nh.oscillation.DeltaMSq12 juno_nh.oscillation.DeltaMSqEE juno_nh.oscillation.SinSq12 \
   # juno_nh.oscillation.SinSq13 \
# -- scan  --lingrid common.Eres_b 0.01 0.06 20 --minimizer min  --verbose --output /tmp/out_ih.hdf5 \
# -- contour --chi2 /tmp/out.hdf5  --minimizer min --plot chi2minplt  --show


#--pullminimizer min
# juno_nh.oscillation.DeltaMSqEE juno_nh.oscillation.SinSq12
# juno_nh.oscillation.SinSq13 \
 # --lingrid common.Eres_b 0.001 0.1 10 --samples-type grid --single --minimizer min \
# -- dataset --name asimov_ih_ih --asimov-data juno_ih/AD1 juno_fake_data/AD1 \
# -- analysis --name testhier_ih -d asimov_ih_ih pull_dm_12 pull_sin_12 \
   # pull_sin_13 -o juno_ih/AD1 \
# -- scan --lingrid juno_ih.oscillation.DeltaMSqEE 2.4e-3 2.6-3 100 \
   # --minimizer min --verbose --output /tmp/out_ih.hdf5 \

# -- dataset --name asimov_ih_ih --asimov-data juno_ih/AD1 juno_fake_data/AD1 \
# -- dataset --name pull_dm_12_ih --pull juno_ih.oscillation.DeltaMSq12 \
# -- dataset --name pull_sin_12_ih --pull juno_ih.oscillation.SinSq12 \
# -- dataset --name pull_sin_13_ih --pull juno_ih.oscillation.SinSq13 \
# -- analysis --name testhier_ih -d asimov_ih_ih pull_dm_12_ih pull_sin_12_ih \
   # pull_sin_13_ih -o juno_ih/AD1 \
# -- chi2 chi2_ih testhier_ih \
# -- minimizer min_ih minuit chi2_ih juno_ih.oscillation.DeltaMSq12 juno_ih.oscillation.DeltaMSqEE juno_ih.oscillation.SinSq12 \
   # juno_ih.oscillation.SinSq13 \
# -- scan  --lingrid juno_ih.oscillation.DeltaMSqEE 2.4e-3 2.6e-3 100 \
# --minimizer min_ih  --verbose --output /tmp/out_ih.hdf5 \

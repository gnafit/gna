#!/bin/sh
python gna -- ns --define common.Eres_b central=0.03 sigma=0.0  \
           --define common.rho_C14 central=1e-17 sigma=1e-19  \
           --define common.CoincidenceWindow central=300 sigma=1 --push common \
-- juno --with-C14 --name juno_ih --ibd first --binning AD1 1 10 700 \
-- ns --value juno_ih.oscillation.Alpha inverted \
-- juno --name juno_nh --with-C14 --ibd first --binning AD1 1 10 700 \
-- dataset --name asimov_ih  --asimov-data juno_nh/AD1 juno_ih/AD1 \
-- analysis --name testhier -d asimov_ih -o juno_nh/AD1 \
-- chi2 chi2 testhier \
-- minimizer min minuit chi2 juno_nh.oscillation.DeltaMSqEE juno_nh.oscillation.SinSq12 \
   juno_nh.oscillation.SinSq13 juno_nh.oscillation.DeltaMSq12 \
-- scan  --lingrid common.CoincidenceWindow 300 1200 40 --minimizer min \
  --verbose --output /tmp/out_c14_coinc.hdf5 \
-- contour --chi2 /tmp/out_c14_coinc.hdf5  --minimizer min --plot chi2minplt \
   --no-shift  --xlabel "Coincidence window, ns" --ylabel "\$ \Delta \chi^2 \$" \
   --output chi2_C14_1e-17_coincidence_window.pdf  --title " \$ \rho_{rel} \$  is fixed to \$ 10^{-17}\$ "
#--pullminimizer min
# juno_nh.oscillation.DeltaMSqEE juno_nh.oscillation.SinSq12
# juno_nh.oscillation.SinSq13 \
 # --lingrid common.Eres_b 0.001 0.1 10 --samples-type grid --single --minimizer min \

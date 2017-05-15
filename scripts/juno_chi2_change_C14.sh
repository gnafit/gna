#!/bin/sh
python gna -- ns --define common.Eres_b central=0.03 sigma=0.0 --push common \
           --define common.rho_C14 central=1e-19 sigma=0.0 --push common \
-- juno --with-C14 --name juno_ih --ibd first --binning AD1 1 10 700 \
-- ns --value juno_ih.oscillation.Alpha inverted \
-- juno --name juno_nh --with-C14 --ibd first --binning AD1 1 10 700 \
-- dataset --name asimov_ih  --asimov-data juno_nh/AD1 juno_ih/AD1 \
-- analysis --name testhier -d asimov_ih -o juno_nh/AD1 \
-- chi2 chi2 testhier \
-- minimizer min minuit chi2 juno_nh.oscillation.DeltaMSqEE juno_nh.oscillation.SinSq12 \
   juno_nh.oscillation.SinSq13 \
-- scan  --lingrid common.rho_C14 1e-19 1e-16 40 --minimizer min \
   --verbose --output /tmp/out.hdf5 \
-- contour --chi2 /tmp/out.hdf5  --minimizer min --plot chi2minplt \
   --no_shift --xlog --xlabel "\$ \rho_{rel} \$" --ylabel "\$ \chi^2 \$" \
   --output chi2_C14.pdf
#--pullminimizer min
# juno_nh.oscillation.DeltaMSqEE juno_nh.oscillation.SinSq12
# juno_nh.oscillation.SinSq13 \
 # --lingrid common.Eres_b 0.001 0.1 10 --samples-type grid --single --minimizer min \

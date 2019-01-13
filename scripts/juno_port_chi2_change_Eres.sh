#!/bin/sh
ipython2 --pdb -- gna -- ns --define common.eres.b central=0.03 sigma=0.0 --push common \
    -- exp  --ns juno_ih juno_port -vv \
    -- ns --value juno_ih.pmns.Alpha inverted \
    -- exp --ns juno_nh juno_port -vv \
    -- ns --replace juno_ih.eres.b common.eres.b \
    # -- ns --replace juno_nh.eres.b common.eres.b \
    # -- repl \
-- dataset --name asimov_ih  --asimov-data juno_nh/AD1 juno_ih/AD1 \
-- analysis --name testhier -d asimov_ih -o juno_nh/AD1 \
-- chi2 chi2 testhier \
-- minimizer min minuit chi2 juno_nh.pmns  \
-s "{juno_nh.pmns.Delta: {fixed: True}}" \
-- scan  --lingrid common.eres.b 0.01 0.06 20 --minimizer min \
   --verbose --output /tmp/out.hdf5 \
-- contour --chi2 /tmp/out.hdf5  --minimizer min --plot chi2minplt \
   --xlabel "E_res" --ylabel "\$\chi^2\$" --no-shift  --output chi2_Eres_change.pdf -s
#--pullminimizer min
# juno_nh.oscillation.DeltaMSqEE juno_nh.oscillation.SinSq12
# juno_nh.oscillation.SinSq13 \
 # --lingrid common.Eres_b 0.001 0.1 10 --samples-type grid --single --minimizer min \
# -- minimizer min minuit chi2 juno_nh.oscillation.DeltaMSqEE \
   # juno_nh.oscillation.SinSq12 juno_nh.oscillation.SinSq13 \
   # --parameters juno_nh.reactors.YJ1.ThermalPower \

#!/bin/sh
python gna -- ns --define common.Eres_b central=0.03 sigma=0.0 --push common \
-- juno --name juno_stand --no-reactor-groups --oscprob standard \
-- juno --name juno_decoh --no-reactor-groups --oscprob decoh \
-- dataset --name asimov_stand --asimov-data juno_decoh/AD1 juno_stand/AD1 \
-- analysis --name testdecoh -d asimov_stand -o juno_decoh/AD1 \
-- chi2 chi2 testdecoh -- minimizer min minuit chi2 juno_decoh.oscillation.DeltaMSqEE juno_decoh.oscillation.SinSq12 juno_decoh.oscillation.SinSq13 \
-- scan --lingrid juno_decoh.oscillation.SigmaDecohRel 0.001 0.01 20 --loggrid juno_decoh.oscillation.SigmaDecohRel 5e-17 1e-15 10  --minimizer min --verbose --output /tmp/out.hdf5 \
-- contour --chi2 /tmp/out.hdf5 --minimizer min --plot chi2minplt --show
#-- scan --lingrid juno_decoh.oscillation.SigmaDecohRel 0.001 0.1 10 --output /tmp/out.hdf5 
#--minimizer min \
#--output /tmp/out.hdf5 \
#-- contour --chi2 /tmp/out.hdf5  --minimizer min 
#--plot chi2minplt   --show
#-- scan --lingrid juno_decoh.oscillation.SigmaDecohRel 0.001 0.01 20 --loggrid juno_decoh.oscillation.SigmaDecohRel 1e-17 1e-15 10  --minimizer min --verbose --output /tmp/out.hdf5 \

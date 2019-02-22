#! /bin/bash

myrun1(){
python gna \
    -- ns \
	--define juno_nh.Qp0 central=0.0065 sigma=1 \
    --push juno_nh \
	-- nl_juno --name juno_nh --with-mine \
    -- ns --pop juno_nh \
	-- ns \
	--define juno_ih.Qp0 central=0.0065 sigma=1 \
    --push juno_ih \
	-- nl_juno --name juno_ih --with-mine \
    -- ns --pop juno_ih \
	-- ns --value juno_ih.oscillation.Alpha inverted \
	-- dataset --name fit_hier_data --asimov-data juno_ih/AD1 juno_nh/AD1 \
	-- analysis --name fit_hier -d fit_hier_data \
	-o juno_ih/AD1 \
	-- chi2 tchi2 fit_hier \
	-- minimizer min minuit tchi2  juno_ih.oscillation.DeltaMSqEE \
        -- contour --no-shift --chi2 /home/cheng/juno/JUNO-SOFT/analysis/gna/mineout.hdf5  --minimizer min --plot chi2minplt \
        --labels "KB scan result (IH hypothesis fit NH fakedata)" --legend upper right\
        --ylim 0 200  \
        --xlabel "KB value" --ylabel "\$ \chi^2 \$"\
        --show
        #-- scan --lingrid juno_ih.Qp0 1.5e-3 11.5e-3 11 \
        #--minimizer min --verbose --output /home/cheng/juno/JUNO-SOFT/analysis/gna/mineout.hdf5 \
}
myrun2(){
python gna \
    -- ns \
	--define juno_nh.Qp0 central=0.0227 sigma=0 \
    --define juno_nh.Qp2 central=1530.1 sigma=0 \
    --define juno_nh.Qp3 central=1.5 sigma=0 \
    --define juno_nh.Qp1 central=0.00015 sigma=0 \
    --push juno_nh \
	-- nl_juno --name juno_nh --with-mine \
    -- ns --pop juno_nh \
	-- ns \
	--define juno_ih.Qp0 central=$1 sigma=0 \
    --define juno_ih.Qp2 central=1530.1 sigma=0 \
    --define juno_ih.Qp3 central=1.5 sigma=0 \
    --define juno_ih.Qp1 central=0.00015 sigma=0 \
    --push juno_ih \
	-- nl_juno --name juno_ih --with-mine \
    -- ns --pop juno_ih \
	-- dataset --name fit_hier_data --asimov-data juno_ih/AD1 juno_nh/AD1 \
	-- analysis --name fit_hier -d fit_hier_data \
	-o juno_ih/AD1 \
	-- chi2 tchi2 fit_hier \
	-- minimizer min minuit tchi2 juno_ih.oscillation.DeltaMSqEE \
    -- fit min



}
	#-- ns --value juno_ih.oscillation.Alpha inverted \

#tests="0.2 0.6 1.0 1.4 1.8"
tests="0.006 0.007 0.008 0.009 0.01  0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.02  0.021 0.022 0.023 0.024 0.025 0.026 0.027 0.028 0.029 0.03  0.031 0.032    0.033 0.034 0.035"
for name in $tests; do
    name2=$(echo $name*0.0065 | bc)
    echo $name $name2
    myrun2  $name
done

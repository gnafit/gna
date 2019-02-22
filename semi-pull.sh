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
	--define juno_nh.Qp0 central=0.0227 sigma=$1 \
    --define juno_nh.Qp2 central=1530.1 sigma=0 \
    --define juno_nh.Qp3 central=1.5 sigma=0 \
    --define juno_nh.Qp1 central=0.00015 sigma=0 \
    --push juno_nh \
	-- nl_juno --name juno_nh --with-mine \
    -- ns --pop juno_nh \
	-- ns \
	--define juno_ih.Qp0 central=0.0227 sigma=$1 \
    --define juno_ih.Qp2 central=1530.1 sigma=0 \
    --define juno_ih.Qp3 central=1.5 sigma=0 \
    --define juno_ih.Qp1 central=0.00015 sigma=0 \
    --push juno_ih \
	-- nl_juno --name juno_ih --with-mine \
    -- ns --pop juno_ih \
	-- ns --value juno_ih.oscillation.Alpha inverted \
	-- dataset --name fit_hier_data --asimov-data juno_ih/AD1 juno_nh/AD1 \
	-- dataset --name pullB0 --pull juno_ih.Qp0 \
	-- analysis --name fit_hier -d fit_hier_data \
	pullB0 \
	-o juno_ih/AD1 \
	juno_ih.Qp0 \
	-- chi2 tchi2 fit_hier \
	-- minimizer min minuit tchi2 juno_ih.oscillation.DeltaMSqEE \
	juno_ih.Qp0 \
    -- fit min


#python gna \
#    -- ns \
#	--define juno_nh.Qp0 central=0.0065 sigma=0 \
#    --push juno_nh \
#	-- nl_juno --name juno_nh --with-mine \
#    -- ns --pop juno_nh \
#	-- ns \
#	--define juno_ih.Qp0 central=0.0065 sigma=$1 \
#    --push juno_ih \
#	-- nl_juno --name juno_ih --with-mine \
#    -- ns --pop juno_ih \
#	-- dataset --name fit_hier_data --asimov-data juno_ih/AD1 juno_nh/AD1 \
#	-- dataset --name pullB0 --pull juno_ih.Qp0 \
#	-- analysis --name fit_hier -d fit_hier_data \
#	pullB0 \
#	-o juno_ih/AD1 \
#	juno_ih.Qp0 \
#	-- chi2 tchi2 fit_hier \
#	-- minimizer min minuit tchi2 juno_ih.oscillation.DeltaMSqEE \
#	juno_ih.Qp0 \
#    -- fit min
#

}

#tests="0.0"
#tests="0.1 0.2 0.6 1.0 1.4 1.8 10"
tests="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 4.0 6.0 8.0 10.0 15.0 20.0"
for name in $tests; do
    name2=$(echo $name*0.0227 | bc)
    echo $name $name2
    myrun2  $name2
done

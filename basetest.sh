#python gna \
#	-- nl_juno --name juno_nh --with-worst \
#	-- nl_juno --name juno_ih \
#	-- ns --value juno_ih.oscillation.Alpha inverted \
#	-- dataset --name fit_hier_data --asimov-data juno_ih/AD1 juno_nh/AD1 \
#	-- analysis --name fit_hier -d fit_hier_data \
#	-o juno_ih/AD1 \
#	-- chi2 tchi2 fit_hier \
#	-- minimizer min minuit tchi2 juno_ih.oscillation.DeltaMSqEE \
#	-- fit min
#
#python gna \
#	-- nl_juno --name juno_nh --with-worst \
#	-- nl_juno --name juno_ih \
#	-- dataset --name fit_hier_data --asimov-data juno_ih/AD1 juno_nh/AD1 \
#	-- analysis --name fit_hier -d fit_hier_data \
#	-o juno_ih/AD1 \
#	-- chi2 tchi2 fit_hier \
#	-- minimizer min minuit tchi2 juno_ih.oscillation.DeltaMSqEE \
#	-- fit min
#

python gna \
	-- nl_juno --name juno_nh --with-worst \
	-- ns \
	--define common.Qp0 central=0.9522 sigma=1.0 \
	--define common.Qp1 central=0.012 sigma=1.0 \
	--define common.Qp2 central=-0.0007 sigma=1.0 \
    --push common \
	-- nl_juno --name juno_ih --with-qua \
	-- ns --value juno_ih.oscillation.Alpha inverted \
	-- dataset --name fit_hier_data --asimov-data juno_ih/AD1 juno_nh/AD1 \
	-- dataset --name pullB0 --pull common.Qp0 \
	-- dataset --name pullB0 --pull common.Qp1 \
	-- dataset --name pullB0 --pull common.Qp2 \
	-- analysis --name fit_hier -d fit_hier_data \
	pullB0 \
	pullB1 \
	pullB2 \
	-o juno_ih/AD1 \
	common.Qp0 \
	common.Qp1 \
	common.Qp2 \
	-- chi2 tchi2 fit_hier \
	-- minimizer min minuit tchi2 juno_ih.oscillation.DeltaMSqEE \
	common.Qp0 \
	common.Qp1 \
	common.Qp2 \
	-- fit min

python gna \
	-- nl_juno --name juno_nh --with-worst \
	-- ns \
	--define common.Qp0 central=0.9522 sigma=1.0 \
	--define common.Qp1 central=0.012 sigma=1.0 \
	--define common.Qp2 central=-0.0007 sigma=1.0 \
    --push common \
	-- nl_juno --name juno_ih --with-qua \
	-- dataset --name fit_hier_data --asimov-data juno_ih/AD1 juno_nh/AD1 \
	-- dataset --name pullB0 --pull common.Qp0 \
	-- dataset --name pullB0 --pull common.Qp1 \
	-- dataset --name pullB0 --pull common.Qp2 \
	-- analysis --name fit_hier -d fit_hier_data \
	pullB0 \
	pullB1 \
	pullB2 \
	-o juno_ih/AD1 \
	common.Qp0 \
	common.Qp1 \
	common.Qp2 \
	-- chi2 tchi2 fit_hier \
	-- minimizer min minuit tchi2 juno_ih.oscillation.DeltaMSqEE \
	common.Qp0 \
	common.Qp1 \
	common.Qp2 \
	-- fit min


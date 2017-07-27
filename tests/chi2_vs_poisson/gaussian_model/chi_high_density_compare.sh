##################################################
#
# Compare plots with and without fluctuations:
#
#
#      Events density -- High
#
##################################################

python ./gna \
-- gaussianpeak --name peak1 --nbins 100 \
-- gaussianpeak --name peak2 --nbins 100 \
-- ns --value peak1.Mu 100000 --value peak1.BackgroundRate 100000  \
      --value peak2.Mu 100000 --value peak2.BackgroundRate 100000  \
 \
-- dataset --name asimov  	  --asimov-poisson     peak1/spectrum peak2/spectrum \
-- dataset --name asimov_data1    --asimov-data        peak1/spectrum peak2/spectrum \
 \
-- analysis --name asimov_analysis 	   --datasets asimov 	     --observables peak1/spectrum \
-- analysis --name asimov_analysis1        --datasets asimov_data1   --observables peak1/spectrum \
 \
-- chi2    chi2_asimov		  asimov_analysis \
-- chi2    chi2_asimov1           asimov_analysis1 \
 \
-- minimizer chi_min		     minuit chi2_asimov 	   peak1.Mu \
-- minimizer chi_min1		     minuit chi2_asimov1 	   peak1.Mu \
-- minimizer empty_minimizer_pois    minuit chi2_asimov \
-- minimizer empty_minimizer_chi     minuit chi2_asimov1 \
 \
-- scan --lingrid peak1.Mu 70000 130000 1000   --minimizer empty_minimizer_chi      --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi.hdf5 \
-- scan --lingrid peak1.Mu 70000 130000 1000   --minimizer empty_minimizer_pois     --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois.hdf5 \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois.hdf5    --minimizer chi_min  --plot  chi2minplt --labels "chi2 aspoisson" 	--legend upper right  \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi.hdf5     --minimizer chi_min1 --plot  chi2minplt --labels "chi2 asdata"   --legend upper right \
	 --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/img/high_den_chi2_compare.png \
#-- spectrum --plot peak1/spectrum --savefig $(pwd)/tests/chi2_vs_poisson/gaussian_model/img/chi2_high_spectrum.png

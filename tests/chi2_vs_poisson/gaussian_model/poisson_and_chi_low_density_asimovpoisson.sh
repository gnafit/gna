##################################################
#
# Compare plots with fluctuations:
#   	  poisson vs. chi2
#
#      Events density -- Low
#
##################################################

python ./gna \
-- gaussianpeak --name peak1 --nbins 100 \
-- gaussianpeak --name peak2 --nbins 100 \
-- ns --value peak1.Mu 10 --value peak1.BackgroundRate 10  \
      --value peak2.Mu 10 --value peak2.BackgroundRate 10  \
 \
-- dataset --name asimov  	  --asimov-poisson     peak1/spectrum peak2/spectrum \
 \
-- analysis --name asimov_analysis 	   --datasets asimov 	     --observables peak1/spectrum \
 \
-- chi2                 chi2_asimov		  asimov_analysis \
-- poisson --ln-approx  pois_asimov		  asimov_analysis \
 \
-- minimizer chi_min		     minuit chi2_asimov 	   peak1.Mu \
-- minimizer pois_min		     minuit pois_asimov 	   peak1.Mu \
-- minimizer empty_minimizer_chi     minuit chi2_asimov \
-- minimizer empty_minimizer_pois     minuit pois_asimov \
 \
-- scan --lingrid peak1.Mu 0 40 100   --minimizer empty_minimizer_chi      --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi.hdf5 \
-- scan --lingrid peak1.Mu 0 40 100   --minimizer empty_minimizer_pois     --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois.hdf5 \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi.hdf5    --minimizer chi_min  --plot  chi2minplt  --labels "chi2" 	--legend upper right  \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois.hdf5   --minimizer pois_min --plot  chi2minplt  --labels "poisson"  --legend upper right \
	 --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/img/low_den_chi_and_poiss_aspois.png


#-- scan	--lingrid peak1.Mu 0 40 100   --minimizer empty_minimizer_pois     --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois.hdf5 \

##################################################
#
# Compare plots without fluctuations:
#   	  poisson vs. chi2
#
#      Events density -- High
#
##################################################

python ./gna \
-- gaussianpeak --name peak1 --nbins 100 \
-- gaussianpeak --name peak2 --nbins 100 \
-- gaussianpeak --name peak3 --nbins 100 \
-- ns --value peak1.Mu 10000 --value peak1.BackgroundRate 10000  \
      --value peak2.Mu 10000 --value peak2.BackgroundRate 10000  \
      --value peak3.Mu 10000 --value peak3.BackgroundRate 10000  \
 \
-- dataset --name asimov  	  --asimov-poisson     peak1/spectrum peak2/spectrum \
-- dataset --name asimov_pois  	  --asimov-poisson     peak3/spectrum peak2/spectrum \
 \
-- analysis --name asimov_analysis 	   --datasets asimov 	     --observables peak1/spectrum \
-- analysis --name poisson_analysis 	   --datasets asimov_pois    --observables peak3/spectrum \
 \
-- chi2    chi2_asimov		  asimov_analysis \
-- poisson pois_asimov		  poisson_analysis \
 \
-- minimizer chi_min		     minuit chi2_asimov 	   peak1.Mu \
-- minimizer pois_min		     minuit pois_asimov 	   peak3.Mu \
-- minimizer empty_minimizer_chi     minuit chi2_asimov \
-- minimizer empty_minimizer_pois     minuit pois_asimov \
 \
-- scan --lingrid peak1.Mu 9000 11000 100   --minimizer empty_minimizer_chi      --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi.hdf5 \
-- scan	--lingrid peak3.Mu 9000 11000 100   --minimizer empty_minimizer_pois     --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois.hdf5 \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi.hdf5    --minimizer chi_min  --plot  chi2minplt --labels "chi2" 	--legend upper right  \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois.hdf5   --minimizer pois_min --plot  chi2minplt --labels "poisson"  --legend upper right \
	 --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/img/high_den_chi_and_poiss_aspois.png

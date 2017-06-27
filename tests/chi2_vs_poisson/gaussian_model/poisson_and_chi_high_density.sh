##################################################
#
# Compare plots without and with fluctuations:
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
-- dataset --name asimov  	     --asimov-data        peak1/spectrum peak2/spectrum \
-- dataset --name asimov_pois  	     --asimov-data        peak3/spectrum peak2/spectrum \
-- dataset --name asimov_ap          --asimov-poisson     peak1/spectrum peak2/spectrum \
-- dataset --name asimov_pois_ap     --asimov-poisson     peak3/spectrum peak2/spectrum \
 \
-- analysis --name asimov_analysis 	      --datasets asimov            --observables peak1/spectrum \
-- analysis --name poisson_analysis 	      --datasets asimov_pois       --observables peak3/spectrum \
-- analysis --name asimov_analysis_ap         --datasets asimov_ap         --observables peak1/spectrum \
-- analysis --name poisson_analysis_ap        --datasets asimov_pois_ap    --observables peak3/spectrum \
 \
-- chi2    chi2_asimov	  	     asimov_analysis \
-- poisson pois_asimov		     poisson_analysis \
-- chi2    chi2_asimov_ap            asimov_analysis_ap \
-- poisson pois_asimov_ap            poisson_analysis_ap \
 \
-- minimizer chi_min		         minuit chi2_asimov 	      peak1.Mu \
-- minimizer pois_min		         minuit pois_asimov 	      peak3.Mu \
-- minimizer chi_min_ap                  minuit chi2_asimov_ap        peak1.Mu \
-- minimizer pois_min_ap                 minuit pois_asimov_ap        peak3.Mu \
-- minimizer empty_minimizer_chi         minuit chi2_asimov \
-- minimizer empty_minimizer_pois        minuit pois_asimov \
-- minimizer empty_minimizer_chi_ap      minuit chi2_asimov_ap \
-- minimizer empty_minimizer_pois_ap     minuit pois_asimov_ap \
 \
-- scan --lingrid peak1.Mu 9000 11000 100   --minimizer empty_minimizer_chi      --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi.hdf5 \
-- scan	--lingrid peak3.Mu 9000 11000 100   --minimizer empty_minimizer_pois     --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois.hdf5 \
-- scan --lingrid peak1.Mu 9000 11000 100   --minimizer empty_minimizer_chi_ap      --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi_ap.hdf5 \
-- scan --lingrid peak3.Mu 9000 11000 100   --minimizer empty_minimizer_pois_ap     --verbose --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois_ap.hdf5 \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi.hdf5    --minimizer chi_min  --plot  chi2minplt --labels "chi2" 	--legend upper right  \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois.hdf5   --minimizer pois_min --plot  chi2minplt --labels "poisson"  --legend upper right \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_chi_ap.hdf5    --minimizer chi_min_ap  --plot  chi2minplt --labels "chi2_ap" --legend upper right \
-- contour --chi2 $(pwd)/tests/chi2_vs_poisson/gaussian_model/tmp/scan_pois_ap.hdf5   --minimizer pois_min_ap --plot  chi2minplt --labels "poisson_ap" --legend upper right \
	 --output $(pwd)/tests/chi2_vs_poisson/gaussian_model/img/high_den_chi_and_poiss.png

#! /bin/sh
python gna gaussianpeak --name two_peaks --nbins 200 --npeaks 2 \
    -- ns --value two_peaks1.E0 2.1 --value two_peaks0.E0 1.9 \
    -- gaussianpeak --name peak --nbins 200 \
    -- ns --value peak.Mu 30 \
    -- dataset --name two_gauss_asimov --asimov-data two_peaks/spectrum peak/spectrum \
    -- analysis --name two_gauss_analysis -d two_gauss_asimov -o two_peaks/spectrum \
    -- chi2 chi2_two_gauss two_gauss_analysis \
    -- minimizer min minuit chi2_two_gauss \
    -- scan --lingrid two_peaks1.Mu 0 300 300 --lingrid two_peaks0.Mu 0 300 300  --verbose --minimizer min --output /tmp/two_gaus.hdf5 \
-- contour --chi2 /tmp/two_gaus.hdf5 --plot chi2ci 1s 2s 3s --minimizer min --show

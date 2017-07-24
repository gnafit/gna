#!/bin/bash

# This file has execution permissions set use it from the root GNA folder as:
# > ./tests/chi2_2d_points/paraboloid_test.sh
#
# Updates:
#   - no 'cd' is required
#   - the output images are written to the 'output' folder
#     (please, keep folders added to the repository clean)
#   - check the existence of 'pdfunite'
#   - produced image file name is printed to stdout


OUTPUT=output/chi2_2d_points
test -d $OUTPUT && rm -r $OUTPUT
mkdir -p $OUTPUT


./tests/chi2_2d_points/testpar_irr.py -irr --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 1  -tol 0.1 -dev 1 -ginf 1 -o $OUTPUT/sp10tol01dev1ginf1_d.pdf

./tests/chi2_2d_points/testpar_irr.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 1  -tol 0.1 -dev 1 -ginf 1 -o $OUTPUT/sp10tol01dev1ginf1.pdf

./tests/chi2_2d_points/testpar_irr.py -irr --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 1  -tol 0.1 -dev 1 -ginf 2 -o $OUTPUT/sp10tol01dev1ginf2_d.pdf

./tests/chi2_2d_points/testpar_irr.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 1  -tol 0.1 -dev 1 -ginf 2 -o $OUTPUT/sp10tol01dev1ginf2.pdf

./tests/chi2_2d_points/testpar_irr.py -irr --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 1  -tol 0.1 -dev 2 -ginf 0.5 -o $OUTPUT/sp10tol01dev2ginf1_d.pdf

./tests/chi2_2d_points/testpar_irr.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 1  -tol 0.1 -dev 2 -ginf 0.5 -o $OUTPUT/sp10tol01dev2ginf1.pdf



if which pdfunite >/dev/null
then
    # test that 'pdfunite' executable exists in the system
    pdfunite $OUTPUT/sp10tol01dev1ginf1_d.pdf $OUTPUT/sp10tol01dev1ginf1.pdf $OUTPUT/sp10tol01dev1ginf2_d.pdf $OUTPUT/sp10tol01dev1ginf2.pdf $OUTPUT/sp10tol01dev2ginf1_d.pdf  $OUTPUT/sp10tol01dev2ginf1.pdf $OUTPUT/dif.pdf
    echo 'Output figure: ' $OUTPUT/dif.pdf
fi


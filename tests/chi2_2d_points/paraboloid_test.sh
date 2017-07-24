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

#sparseness
: './tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 1  -tol 0.1 -dev 1 -ginf 1. -o $OUTPUT/sp1tol01dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 5  -tol 0.1 -dev 1 -ginf 1. -o $OUTPUT/sp5tol01dev1ging1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 1. -o $OUTPUT/sp10tol01dev1ginf1.pdf

pdfunite $OUTPUT/sp5tol01dev1ging1.pdf $OUTPUT/sp1tol01dev1ginf1.pdf $OUTPUT/sp10tol01dev1ginf1.pdf $OUTPUT/dif_sp.pdf
'
#tolerance
: './tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.0 -dev 1 -ginf 1. -o $OUTPUT/sp2tol0dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.2 -dev 1 -ginf 1. -o $OUTPUT/sp2tol20dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.5 -dev 1 -ginf 1. -o $OUTPUT/sp2tol05dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 1.0 -dev 1 -ginf 1. -o $OUTPUT/sp2tol1dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 2.0 -dev 1 -ginf 1. -o $OUTPUT/sp2tol2dev1ginf1.pdf

pdfunite $OUTPUT/sp2tol0dev1ginf1.pdf $OUTPUT/sp2tol20dev1ginf1.pdf $OUTPUT/sp2tol05dev1ginf1.pdf $OUTPUT/sp2tol1dev1ginf1.pdf $OUTPUT/sp2tol2dev1ginf1.pdf $OUTPUT/dif_tol.pdf
'
#deviation
: './tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.1 -dev 1 -ginf 1. -o $OUTPUT/sp2tol01dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.1 -dev 2 -ginf 1. -o $OUTPUT/sp2tol01dev2ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.1 -dev 5 -ginf 1. -o $OUTPUT/sp2tol01dev5ginf1.pdf

pdfunite $OUTPUT/sp2tol01dev1ginf1.pdf $OUTPUT/sp2tol01dev2ginf1.pdf $OUTPUT/sp2tol01dev5ginf1.pdf $OUTPUT/dif_dev.pdf
'
#gradient influence
./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 0.5 -o $OUTPUT/sp10tol01dev1ginf05.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 1. -o $OUTPUT/sp10tol01dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 1.5 -o $OUTPUT/sp10tol01dev1ginf15.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 2. -o $OUTPUT/sp10tol01dev1ginf2.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 3. -o $OUTPUT/sp10tol01dev1ginf3.pdf

if which pdfunite >/dev/null
then
    # test that 'pdfunite' executable exists in the system
    pdfunite $OUTPUT/sp10tol01dev1ginf05.pdf $OUTPUT/sp10tol01dev1ginf1.pdf $OUTPUT/sp10tol01dev1ginf15.pdf $OUTPUT/sp10tol01dev1ginf2.pdf $OUTPUT/sp10tol01dev1ginf3.pdf $OUTPUT/dif_ginf.pdf
    echo 'Output figure: ' $OUTPUT/dif_ginf.pdf
fi


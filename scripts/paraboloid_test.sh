cd ../
rm -r tests/chi2_2d_points/imgs/
mkdir tests/chi2_2d_points/imgs/


#sparseness
: './tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 1  -tol 0.1 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp1tol01dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 5  -tol 0.1 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp5tol01dev1ging1.pdf 

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp10tol01dev1ginf1.pdf

pdfunite tests/chi2_2d_points/imgs/sp5tol01dev1ging1.pdf tests/chi2_2d_points/imgs/sp1tol01dev1ginf1.pdf tests/chi2_2d_points/imgs/sp10tol01dev1ginf1.pdf tests/chi2_2d_points/imgs/dif_sp.pdf
'
#tolerance
: './tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.0 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp2tol0dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.2 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp2tol20dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.5 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp2tol05dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 1.0 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp2tol1dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 2.0 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp2tol2dev1ginf1.pdf

pdfunite tests/chi2_2d_points/imgs/sp2tol0dev1ginf1.pdf tests/chi2_2d_points/imgs/sp2tol20dev1ginf1.pdf tests/chi2_2d_points/imgs/sp2tol05dev1ginf1.pdf tests/chi2_2d_points/imgs/sp2tol1dev1ginf1.pdf tests/chi2_2d_points/imgs/sp2tol2dev1ginf1.pdf tests/chi2_2d_points/imgs/dif_tol.pdf
'
#deviation
: './tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.1 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp2tol01dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.1 -dev 2 -ginf 1. -o tests/chi2_2d_points/imgs/sp2tol01dev2ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 2  -tol 0.1 -dev 5 -ginf 1. -o tests/chi2_2d_points/imgs/sp2tol01dev5ginf1.pdf

pdfunite tests/chi2_2d_points/imgs/sp2tol01dev1ginf1.pdf tests/chi2_2d_points/imgs/sp2tol01dev2ginf1.pdf tests/chi2_2d_points/imgs/sp2tol01dev5ginf1.pdf tests/chi2_2d_points/imgs/dif_dev.pdf
'
#gradient influence
./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 0.5 -o tests/chi2_2d_points/imgs/sp10tol01dev1ginf05.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 1. -o tests/chi2_2d_points/imgs/sp10tol01dev1ginf1.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 1.5 -o tests/chi2_2d_points/imgs/sp10tol01dev1ginf15.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 2. -o tests/chi2_2d_points/imgs/sp10tol01dev1ginf2.pdf

./tests/chi2_2d_points/testpar.py --xlinspace -2.0 6 501 --ylinspace -4.0 6 501 -sp 10  -tol 0.1 -dev 1 -ginf 3. -o tests/chi2_2d_points/imgs/sp10tol01dev1ginf3.pdf

pdfunite tests/chi2_2d_points/imgs/sp10tol01dev1ginf05.pdf tests/chi2_2d_points/imgs/sp10tol01dev1ginf1.pdf tests/chi2_2d_points/imgs/sp10tol01dev1ginf15.pdf tests/chi2_2d_points/imgs/sp10tol01dev1ginf2.pdf tests/chi2_2d_points/imgs/sp10tol01dev1ginf3.pdf tests/chi2_2d_points/imgs/dif_ginf.pdf

cd scripts/

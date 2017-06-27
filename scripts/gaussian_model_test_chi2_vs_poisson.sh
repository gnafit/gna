cd ..
rm -r tests/chi2_vs_poisson/gaussian_model/tmp
rm -r tests/chi2_vs_poisson/gaussian_model/img
mkdir tests/chi2_vs_poisson/gaussian_model/tmp
mkdir tests/chi2_vs_poisson/gaussian_model/img
. tests/chi2_vs_poisson/gaussian_model/chi_high_density_compare.sh
. tests/chi2_vs_poisson/gaussian_model/poisson_and_chi_high_density.sh
. tests/chi2_vs_poisson/gaussian_model/poisson_high_density_compare.sh
. tests/chi2_vs_poisson/gaussian_model/chi_low_density_compare.sh
. tests/chi2_vs_poisson/gaussian_model/poisson_low_density_compare.sh
. tests/chi2_vs_poisson/gaussian_model/chi_middle_density_compare.sh
. tests/chi2_vs_poisson/gaussian_model/poisson_and_chi_low_density_asimovpoisson.sh # doesn't work
. tests/chi2_vs_poisson/gaussian_model/poisson_middle_density_compare.sh
. tests/chi2_vs_poisson/gaussian_model/poisson_and_chi_middle_density_asimovpoisson.sh
rm -r tests/chi2_vs_poisson/gaussian_model/tmp
cd scripts/

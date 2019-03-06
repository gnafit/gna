#include "PolynomialNonlinearity.hh"
#include "TypesFunctions.hh"
#include "fmt/ostream.h"


PolynomialNonlinearity::PolynomialNonlinearity(int poly_order): m_poly_order{poly_order}, m_coeffs(m_poly_order) {
    this->transformation_("nl_edges")
        .input("old_edges")
        .output("new_edges")
        .types(TypesFunctions::pass<0>)
        .func(&PolynomialNonlinearity::computeNewBins);

    for (int i = 0; i < m_poly_order; ++i) {
        variable_(&m_coeffs.at(i), fmt::format("coeff_{}", i));
    }
}

 /* Use the following formula to shift bin edges:
  * Enew/Eold = coeff_0 + coeff_1 * Eold + .. coeff_n * Eold^n  */
void PolynomialNonlinearity::computeNewBins(FunctionArgs& fargs) noexcept {
    const auto& old_edges = fargs.args[0].x;
    auto& new_edges = fargs.rets[0].x;

    Eigen::ArrayXd accumulator = Eigen::ArrayXd::Ones(new_edges.size());
    for (const auto& coeff: m_coeffs) {
        new_edges += coeff*accumulator;
        accumulator *= old_edges;
    }
}

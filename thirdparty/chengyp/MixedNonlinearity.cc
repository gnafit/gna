#include "MixedNonlinearity.hh"
#include "TypesFunctions.hh"

MixedNonlinearity::MixedNonlinearity() {
    this->transformation_("ExpNL")
        .input("old_bins")
        .output("bins_after_nl")
        .types(TypesFunctions::pass<0>)
        .func(&MixedNonlinearity::computeNewBins);

    variable_(&m_alpha, "Exp_p0");
    variable_(&m_beta, "Exp_p1");
}

void MixedNonlinearity::computeNewBins(FunctionArgs& fargs) noexcept {
    const auto& old_bins = fargs.args[0].x;
    auto& bins_after_nl = fargs.rets[0].x;

    bins_after_nl = (m_alpha + 1)*((1 + m_beta*exp(-0.2*old_bins)).inverse()) * old_bins;
}

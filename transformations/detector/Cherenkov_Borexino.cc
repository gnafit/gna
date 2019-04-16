#include "Cherenkov_Borexino.hh"
#include <algorithm>

Cherenkov_Borexino::Cherenkov_Borexino() {
    variable_(&p0, "p0");
    variable_(&p1, "p1");
    variable_(&p2, "p2");
    variable_(&p3, "p3");
    variable_(&p4, "p4");
    variable_(&E_0, "E_0");
    transformation_("cherenkov")
        .input("energy")
        .output("ch_npe")
        .types(TypesFunctions::passAll)
        .func(&Cherenkov_Borexino::calc_Cherenkov);
}

void Cherenkov_Borexino::calc_Cherenkov(FunctionArgs fargs) {
    auto* energy_buf = fargs.args[0].buffer;
    auto size = fargs.args[0].x.size();
    auto* end_of_buf = energy_buf + size;

    // fill with 0. below threshold
    auto* after_treshold = std::lower_bound(energy_buf, end_of_buf, E_0);
    std::ptrdiff_t num_under_tresh = after_treshold - energy_buf;
    std::ptrdiff_t num_above_tresh = end_of_buf - after_treshold;

    auto& ret = fargs.rets[0].x;
    ret.head(num_under_tresh) = 0.;

    const Eigen::Map<Eigen::ArrayXd> energy(after_treshold, num_above_tresh);
    Eigen::ArrayXd X  = (1.0 + energy/E_0).log();
    Eigen::ArrayXd X2 = X*X;
    auto out = ret.tail(num_above_tresh);
    out=(p0.value() + p1.value()*X + p2.value()*X2 + p3.value()*(X*X2))*(1.0+p4*energy);
    out = out.unaryExpr([](double x){ return x>0.0 ? x : 0.0; });
}

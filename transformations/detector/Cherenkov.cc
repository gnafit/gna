#include "Cherenkov.hh"
#include <algorithm>
#include "fmt/ostream.h"

Cherenkov::Cherenkov() {
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
        .func(&Cherenkov::calc_Cherenkov);
}

void Cherenkov::calc_Cherenkov(FunctionArgs fargs) {
    auto* energy_buf = fargs.args[0].buffer;
    auto size = fargs.args[0].x.size();
    auto* end_of_buf = energy_buf + size;
    //
    // fill with 0. below threshold
    auto* after_treshold = std::lower_bound(energy_buf, end_of_buf, E_0);
    std::ptrdiff_t num_under_tresh = after_treshold - energy_buf;
    std::ptrdiff_t num_above_tresh = end_of_buf - after_treshold;

    fargs.rets[0].x.head(num_under_tresh) = 0.;

    Eigen::ArrayXd energy = Eigen::Map<Eigen::ArrayXd>(after_treshold, num_above_tresh);
    auto x = log(1+energy/E_0);
    auto cherenkov_contrib = (p0 + p1*x + p2*x.square() + p3*x.cube())*(1+p4*energy);
    fargs.rets[0].x.tail(num_above_tresh) = cherenkov_contrib;
}

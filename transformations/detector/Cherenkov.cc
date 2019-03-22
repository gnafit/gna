#include "Cherenkov.hh"

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
    auto energy = fargs.args[0].x;
    auto x = log(1+energy/E_0);
    auto cherenkov_contrib = (p0 + p1*x + p2*x.square() + p3*x.cube())*(1+p4*energy);
    fargs.rets[0].x = cherenkov_contrib;
}

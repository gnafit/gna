#include "Chi2CNPStat.hh"
#include <Eigen/Core>

Chi2CNPStat::Chi2CNPStat() {
    transformation_("chi2")
        .output("chi2")
        .types(&Chi2CNPStat::checkTypes)
        .func(&Chi2CNPStat::calcChi2CNPStat)
        ;

    m_transform = t_["chi2"];
}

void Chi2CNPStat::add(SingleOutput &theory, SingleOutput &data) {
    auto chi2 = transformations["chi2"];
    chi2.input(theory);
    chi2.input(data);
}

void Chi2CNPStat::checkTypes(TypesFunctionArgs fargs) {
    auto& args=fargs.args;
    auto& rets=fargs.rets;
    if (args.size()%2 != 0) {
        throw args.undefined();
    }
    for (size_t i = 0; i < args.size(); i+=2) {
        if (args[i+1].shape != args[i+0].shape) {
            throw rets.error(rets[0], "data and theory have different shape");
        }
    }
    rets[0] = DataType().points().shape(1);
}

void Chi2CNPStat::calcChi2CNPStat(FunctionArgs fargs) {
    /***************************************************************************
     * χ² = (1/3) Σᵢ [(1/dataᵢ+2/theoryᵢ)·(theoryᵢ-dataᵢ)²]
     ****************************************************************************/
    auto& args=fargs.args;

    double res=0.0;
    for (size_t i = 0; i < args.size(); i+=2) {
        auto& theory=args[i+0].arr;
        auto& data=args[i+1].arr;
        res += ((data - theory).square()*(1.0/data + 2.0/theory)).sum();
    }
    fargs.rets[0].arr(0) = res/3.0;
}


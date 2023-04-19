#include <boost/math/constants/constants.hpp>
#include "EnergyResolutionSigmaRelABC.hh"
#include "TypesFunctions.hh"

EnergyResolutionSigmaRelABC::EnergyResolutionSigmaRelABC(const std::vector<std::string>& pars) {
    if(pars.size()!=3u){
        throw std::runtime_error("Energy resolution sigma should have exactly 3 parameters");
    }
    variable_(&m_a, pars[0]);
    variable_(&m_b, pars[1]);
    variable_(&m_c, pars[2]);

    transformation_("sigma")
        .input("energy")
        .output("sigma")
        .types(TypesFunctions::pass<0,0>)
        .func(&EnergyResolutionSigmaRelABC::calcSigma);
}

void EnergyResolutionSigmaRelABC::calcSigma(FunctionArgs& fargs) {
    auto& energy=fargs.args[0].x;
    auto& ret=fargs.rets[0].x;

    ret=(pow(m_a, 2)+ pow(m_b, 2)/energy + pow(m_c, 2)/energy.square()).sqrt();
}


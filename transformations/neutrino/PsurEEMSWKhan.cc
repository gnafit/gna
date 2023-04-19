#include "PsurEEMSWKhan.hh"
#include "TypeClasses.hh"
#include <Eigen/src/Core/Array.h>
#include <stdexcept>
using namespace TypeClasses;

#include "Units.hh"
#include "DensityConversion.hh"
using NeutrinoUnits::oscprobArgumentFactor;
using NeutrinoUnits::oscprobArgumentFactor;
using NeutrinoUnits::aem;
using NeutrinoUnits::g;
using NeutrinoUnits::cm3;
using NeutrinoUnits::km;
using NeutrinoUnits::eV2;
using NeutrinoUnits::MeV;

PsurEEMSWKhan::PsurEEMSWKhan(std::string l_name, std::string rho_name, std::vector<std::string> dmnames):
PsurEEMSWKhan(NeutrinoUnits::conversion::electron_fraction, l_name, rho_name, dmnames)
{

}

PsurEEMSWKhan::PsurEEMSWKhan(double electron_fraction, std::string l_name, std::string rho_name, std::vector<std::string> dmnames)
:  m_density_const(electron_fraction/aem*g/cm3*(-2)*std::sqrt(2)*NeutrinoUnits::Gf), m_dm(3)
{
    switch(dmnames.size()){
        case 0u:
            dmnames={"DeltaMSq12", "DeltaMSq13", "DeltaMSq23"};
        case 3u:
            for (int i = 0; i < 3; ++i) {
                variable_(&m_dm[i], dmnames.at(i));
            }
            break;
        default:
            throw std::runtime_error("OscProb3: expects 3 mass splitting names");
    }

    variable_(&m_L, l_name);
    variable_(&m_rho, rho_name);
    variable_(&m_Alpha, "Alpha");
    variable_(&m_dm_ee, "DeltaMSqEE");
    variable_(&m_sinSq13, "SinSq13");
    variable_(&m_cosSq13, "CosSq13");
    variable_(&m_sinSq12, "SinSq12");
    variable_(&m_cosSq12, "CosSq12");
    variable_(&m_cosDouble12, "CosDouble12");

    this->transformation_("oscprob")
        .input("Enu")
        .output("oscprob")
        .types(new PassTypeT<double>(0, {0,-1}))
        .types(&PsurEEMSWKhan::types)
        .func(&PsurEEMSWKhan::calcOscProb)
        ;

}

void PsurEEMSWKhan::types(typename PsurEEMSWKhan::TypesFunctionArgs& fargs) {
    auto size=fargs.args[0].size();
    m_a.resize(size);
    m_common_21.resize(size);
    m_common_21_squared.resize(size);
    m_common_3.resize(size);

    m_sinSq13_mod.resize(size);
    m_sinSq12_mod.resize(size);
    m_dm21_mod.resize(size);
    m_dm31_mod.resize(size);
    m_dm32_mod.resize(size);
}

void PsurEEMSWKhan::calcOscProb(typename PsurEEMSWKhan::FunctionArgs& fargs) {
    auto &Enu = fargs.args[0].x; // [MeV], implicit, to be treated later in formulas for a and L/E
    auto& ret = fargs.rets[0].x;

    //
    // Mass splittings
    //
    const double dm_21 = m_dm[0].value()*eV2;
    const double alpha = m_Alpha.value();
    const double dm_31 = alpha*m_dm[1].value()*eV2;
    const double dm_32 = alpha*m_dm[2].value()*eV2;
    const double dm_ee = alpha*m_dm_ee.value()*eV2;

    //
    // Mixing angles and parameters
    //
    const double cosSq13=m_cosSq13.value();
    const double sinSq13=m_sinSq13.value();

    const double sinSq12=m_sinSq12.value();
    const double cosSq12=m_cosSq12.value();
    const double scSq12=sinSq12*cosSq12;
    const double cosDouble12=m_cosDouble12.value();

    //
    // Common constants
    //
    // L, converted, quarter = L[km]/4, MeV for 1/E
    const double Lcq = 0.25*m_L.value()*km/MeV;
    // A: eq.5: (-) for antineutrino, MeV for Energy
    const double msw_factor = m_density_const*m_rho.value();
    // a: eq (5) at page 2
    m_a = msw_factor*Enu;
    // from eq (15-20) at page 3
    m_common_21 = (cosSq13/dm_21)*m_a;
    m_common_21_squared = m_common_21.square();

    //
    // Modified oscillation parameters
    //
    // Eq. 17
    m_sinSq13_mod = sinSq13 + (2*sinSq13*cosSq13/dm_ee)*m_a;
    // Eq. 15
    m_sinSq12_mod = sinSq12 + (2*scSq12)*m_common_21 + (3*scSq12*cosDouble12)*m_common_21_squared;

    // Eq. 16
    m_dm21_mod = dm_21 - (dm_21*cosDouble12)*m_common_21 + (2*dm_21*scSq12)*m_common_21_squared;

    // Eqs. 19, 20
    m_common_3 = (scSq12*cosSq13)*m_common_21;
    m_dm31_mod = dm_31 - m_a*((cosSq12*cosSq13-sinSq13) - m_common_3);
    m_dm32_mod = dm_32 - m_a*((sinSq12*cosSq13-sinSq13) + m_common_3);

    //
    // Survival probability, eq. 4
    // Pee = 1 - cos⁴θ₁₃ sin²2θ₁₂ sin²Δ₂₁ - sin²2θ₁₃ (cos²θ₁₂ sin²Δ₃₁ + sin²θ₁₂ sin²Δ₃₂) [all parameters are modified]
    // cos⁴θ  = (1-sin²θ)²
    // sin²2θ = 4 sin²θ cos²θ
    // sin²2θ = 4 sin²θ (1-sin²θ)
    //  = 1 - 4   |--------cos⁴θ₁₃---------|   |--------sin²2θ₁₂/4-----------|
    ret = 1 - 4*( (1-m_sinSq13_mod).square() * m_sinSq12_mod*(1-m_sinSq12_mod) *
    //                                          |--------sin² Δm²₂₁ L/(4E)------------|
                                                (Lcq*(m_dm21_mod/Enu)).sin().square() ) -
    //        |-------------sin²2θ₁₃-----------|
              4*(m_sinSq13_mod*(1-m_sinSq13_mod) * (
    //                                           |----cos²θ₁₂----|   |--------sin² Δm²₃₁ L/(4E)----------|
                                                 (1-m_sinSq12_mod) * (Lcq*(m_dm31_mod/Enu)).sin().square() +
    //                                           |--sin²θ₁₂--|   |--------sin² Δm²₃₂ L/(4E)----------|
                                                 m_sinSq12_mod * (Lcq*(m_dm32_mod/Enu)).sin().square()
                                                 )
                )
        ;
}


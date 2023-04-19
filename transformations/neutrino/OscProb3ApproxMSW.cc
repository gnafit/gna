#include "OscProb3ApproxMSW.hh"
#include "TypeClasses.hh"
#include <Eigen/src/Core/Array.h>
#include <stdexcept>
using namespace TypeClasses;

#include "Units.hh"
#include "DensityConversion.hh"
using NeutrinoUnits::oscprobArgumentFactor;
static const double A = 2*std::sqrt(2)* NeutrinoUnits::Gf * NeutrinoUnits::conversion::density_to_MeV;

template <int N> double pow(double a) { return pow<N-1>(a)*a; }
template<> double pow<0>(double a) { return 1; }
template<> double pow<1>(double a) { return a; }

OscProb3ApproxMSW::OscProb3ApproxMSW(std::string l_name, std::string rho_name, std::vector<std::string> dmnames)
:  m_dm(3)
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
    variable_(&m_sinSqDouble12, "SinSqDouble12");
    variable_(&m_sinSqDouble13, "SinSqDouble13");
    variable_(&m_cosDouble12, "CosDouble12");
    variable_(&m_cosSq13, "CosSq13");

    this->transformation_("oscprob")
        .input("Enu")
        .output("oscprob")
        .depends(m_L, m_dm[0], m_dm[1], m_dm[2], m_rho,
                 m_sinSqDouble12, m_sinSqDouble13,
                 m_cosDouble12, m_cosSq13)
        .types(new PassTypeT<double>(0, {0,-1}), [](TypesFunctionArgs& fargs){
                fargs.ints[0] = DataType().points().shape(fargs.args[0].size());
                fargs.ints[1] = DataType().points().shape(fargs.args[0].size());
                fargs.ints[2] = DataType().points().shape(fargs.args[0].size());
                fargs.ints[3] = DataType().points().shape(fargs.args[0].size());
                fargs.ints[4] = DataType().points().shape(fargs.args[0].size());
                fargs.ints[5] = DataType().points().shape(fargs.args[0].size());
                fargs.ints[6] = DataType().points().shape(fargs.args[0].size());
                fargs.ints[7] = DataType().points().shape(fargs.args[0].size());
                fargs.ints[8] = DataType().points().shape(fargs.args[0].size());
                })
        .func(&OscProb3ApproxMSW::calcOscProb)
        ;

}

void OscProb3ApproxMSW::calcOscProb(typename OscProb3ApproxMSW::FunctionArgs& fargs) {
    auto &Enu = fargs.args[0].x;
    auto& ret = fargs.rets[0].x;
    const double dm_21 = m_dm[0].value();
    // need conversion from ev^2 to MeV^2
    fargs.ints[0].x = (m_cosDouble12.value()+(A*m_rho.value()/dm_21*1e12)*Enu).square() + m_sinSqDouble12.value();
    const Eigen::ArrayXd& common_part = fargs.ints[0].x;

    fargs.ints[1].x = dm_21*common_part.sqrt();
    const Eigen::ArrayXd& dm12_MSW = fargs.ints[1].x;

    fargs.ints[2].x = m_sinSqDouble12 * common_part.inverse();
    const Eigen::ArrayXd& sinSqDouble12_MSW = fargs.ints[2].x;

    fargs.ints[3].x = 1 - sinSqDouble12_MSW;
    const Eigen::ArrayXd& cosSqDouble12_MSW = fargs.ints[3].x;

    fargs.ints[4].x = (1 + cosSqDouble12_MSW.sqrt())/2;
    const Eigen::ArrayXd& cosSq12_MSW = fargs.ints[4].x;

    fargs.ints[5].x = (1 - cosSqDouble12_MSW.sqrt())/2;
    const Eigen::ArrayXd& sinSq12_MSW =  fargs.ints[5].x;

    fargs.ints[6].x = sin((m_dm[1].value()*oscprobArgumentFactor*m_L.value()*0.25)*Enu.inverse()).square();
    const Eigen::ArrayXd& comp13 = fargs.ints[6].x;

    fargs.ints[7].x = sin((m_dm[2].value()*oscprobArgumentFactor*m_L.value()*0.25)*Enu.inverse()).square();
    const Eigen::ArrayXd& comp23 = fargs.ints[7].x;

    fargs.ints[8].x = sin((dm12_MSW*oscprobArgumentFactor*m_L.value()*0.25)*Enu.inverse()).square();
    const Eigen::ArrayXd& comp12_MSW = fargs.ints[8].x;

    ret = 1;
    ret -= m_sinSqDouble13.value()*(cosSq12_MSW*comp13 + sinSq12_MSW*comp23);
    ret -= pow<2>(m_cosSq13.value())*sinSqDouble12_MSW*comp12_MSW;
}


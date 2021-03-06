#include <boost/math/constants/constants.hpp>
#include "TypesFunctions.hh"

#include <TMath.h>

#include "ReactorNorm.hh"

const double pi = boost::math::constants::pi<double>();

ReactorNormAbsolute::ReactorNormAbsolute(const std::vector<std::string> &isonames)
{
  variable_(&m_norm, "Norm");
  auto norm = transformation_("isotopes")
    .types(TypesFunctions::ifSame, [](TypesFunctionArgs fargs) {
        for (size_t i = 0; i < fargs.rets.size(); ++i) {
          fargs.rets[i] = DataType().points().shape(1);
        }
      })
    .func([](ReactorNormAbsolute *obj, FunctionArgs fargs) {
        auto& args=fargs.args;
        auto& rets=fargs.rets;
        for (size_t i = 0; i < rets.size(); ++i) {
          rets[i].x[0] = obj->m_norm*args[i].x.sum();
        }
      })
  ;
  for (const std::string &isoname: isonames) {
    norm.input("fission_fraction_"+isoname);
    norm.output("norm_"+isoname);
  }
}

ReactorNorm::ReactorNorm(const std::vector<std::string> &isonames)
  : m_ePerFission(isonames.size())
{
  variable_(&m_thermalPower, "ThermalPower");
  for (size_t i = 0; i < isonames.size(); ++i) {
    variable_(&m_ePerFission[i], "EnergyPerFission_"+isonames[i]);
  }
  variable_(&m_targetProtons, "TargetProtons");
  variable_(&m_L, "L");
  auto norm = transformation_("isotopes")
    .types(TypesFunctions::ifSame, [](TypesFunctionArgs fargs) {
        for (size_t i = 0; i < fargs.rets.size(); ++i) {
          fargs.rets[i] = DataType().points().shape(1);
        }
      })
    .func(&ReactorNorm::calcIsotopeNorms)
  ;
  for (const std::string &isoname: isonames) {
    norm.input("fission_fraction_"+isoname);
    norm.output("norm_"+isoname);
  }
  norm.input("livetime");
  norm.input("power_rate");
}

void ReactorNorm::calcIsotopeNorms(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  const auto &livetime = args[rets.size()+0].x;
  const auto &power_rate = args[rets.size()+1].x;
  static double conversionFactor = 1.0e-7/TMath::Qe();
  auto distanceWeight = (conversionFactor / (4*pi*std::pow(m_L, 2)));
  auto coeff = m_targetProtons*m_thermalPower*distanceWeight;

  double ePerFission_avg=0.0;
  for (size_t i = 0; i < rets.size(); ++i) {
    ePerFission_avg+=args[i].x(0)*m_ePerFission[i].value();
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    rets[i].x[0] = (coeff*livetime*power_rate*args[i].x/ePerFission_avg).sum();
  }
}

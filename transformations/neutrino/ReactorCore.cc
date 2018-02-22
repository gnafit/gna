#include "ReactorCore.hh"
#include "TypesFunctions.hh"

ReactorCore::ReactorCore(const std::vector<std::string> &isonames)
  : m_ePerFission(isonames.size())
{
  variable_(&m_nominalThermalPower, "NominalThermalPower");
  for (size_t i = 0; i < isonames.size(); ++i) {
    variable_(&m_ePerFission[i], "EnergyPerFission_"+isonames[i]);
  }
  auto reactor = transformation_("reactor")
    .input("efficiency")
    .types(TypesFunctions::ifSame, [](Atypes args, Rtypes rets) {
        for (size_t i = 0; i < rets.size(); ++i) {
          rets[i] = args[0];
        }
      })
    .func(&ReactorCore::calcRates)
  ;
  for (const std::string &isoname: isonames) {
    reactor.input("fraction_"+isoname);
    reactor.output("rate_"+isoname);
  }
}

void ReactorCore::calcRates(Args args, Rets rets) {
  for (size_t i = 0; i < rets.size(); ++i) {
    rets[i].x = m_nominalThermalPower*args[0].x*args[1+i].x/m_ePerFission[i];
  }
}

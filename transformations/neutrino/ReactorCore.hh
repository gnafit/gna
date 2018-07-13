#pragma once

#include <vector>
#include <string>

#include "GNAObject.hh"

class ReactorCore: public GNAObject,
                   public TransformationBind<ReactorCore> {
public:
  ReactorCore(const std::vector<std::string> &isonames);
protected:
  void calcRates(FunctionArgs fargs);

  variable<double> m_nominalThermalPower;
  std::vector<variable<double>> m_ePerFission;
};

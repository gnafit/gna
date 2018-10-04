#ifndef REACTORCORE_H
#define REACTORCORE_H

#include <vector>
#include <string>

#include "GNAObject.hh"

class ReactorCore: public GNAObject,
                   public TransformationBind<ReactorCore> {
public:
  ReactorCore(const std::vector<std::string> &isonames);
protected:
  void calcRates(Args args, Rets rets);

  variable<double> m_nominalThermalPower;
  std::vector<variable<double>> m_ePerFission;
};

#endif // REACTORCORE_H

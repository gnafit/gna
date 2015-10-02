#ifndef PDGVARIABLES_H
#define PDGVARIABLES_H

#include <string>
#include <vector>

#include "ParametersGroup.hh"

class PDGVariables: public ParametersGroup {
public:
  PDGVariables(GNAObject *parent)
    : ParametersGroup(parent, fields(), {}) { }
  PDGVariables(GNAObject *parent, std::vector<std::string> params)
    : PDGVariables(parent)
    { initFields(params); }

  variable<double> NeutronLifeTime;
  variable<double> ProtonMass;
  variable<double> NeutronMass;
  variable<double> ElectronMass;
protected:
  Fields fields() {
    Fields allvars = {
      {"NeutronLifeTime", &NeutronLifeTime},
      {"ProtonMass", &ProtonMass},
      {"NeutronMass", &NeutronMass},
      {"ElectronMass", &ElectronMass},
    };
    return allvars;
  }
};

#endif // PDGVARIABLES_H

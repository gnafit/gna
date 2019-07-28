#pragma once

#include <string>
#include <vector>

#include "ParametersGroup.hh"

using ParametersGroup = GNA::GNAObjectTemplates::ParametersGroupT<double>;

class PDGVariables: public ParametersGroup {
public:
  PDGVariables(GNAObject *parent)
    : ParametersGroup(parent, fields()) { }
  PDGVariables(GNAObject *parent, std::vector<std::string> params)
    : PDGVariables(parent)
    { initFields(params); }

  variable<double> NeutronLifeTime;
  variable<double> ProtonMass;
  variable<double> NeutronMass;
  variable<double> ElectronMass;
protected:
  Fields fields() {
    return Fields()
      .add(&NeutronLifeTime, "NeutronLifeTime")
      .add(&ProtonMass, "ProtonMass")
      .add(&NeutronMass, "NeutronMass")
      .add(&ElectronMass, "ElectronMass")
    ;
  }
};

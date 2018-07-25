#pragma once

#include <string>

#include "GNAObject.hh"
#include "TypesFunctions.hh"

class EvisToEe: public GNASingleObject,
                public TransformationBind<EvisToEe> {
public:
  EvisToEe() {
    variable_(&m_me, "ElectronMass");
    transformation_("Ee")
      .input("Evis")
      .output("Ee")
      .types(TypesFunctions::passAll)
      .func([](EvisToEe *obj, FunctionArgs fargs) {
          fargs.rets[0].x = fargs.args[0].x - obj->m_me;
        });
  }
protected:
  variable<double> m_me;
};

#ifndef EVISTOEE_H
#define EVISTOEE_H

#include <string>

#include "GNAObject.hh"

class EvisToEe: public GNASingleObject,
                public Transformation<EvisToEe> {
public:
  EvisToEe() {
    variable_(&m_me, "ElectronMass");
    transformation_(this, "Ee")
      .input("Evis")
      .output("Ee")
      .types(Atypes::passAll)
      .func([](EvisToEe *obj, Args args, Rets rets) {
          rets[0].x = args[0].x - obj->m_me;
        });
  }
protected:
  variable<double> m_me;
};

#endif // EVISTOEE_H

#pragma once

#include <string>

#include "GNAObject.hh"

class FillLike: public GNASingleObject,
                public TransformationBind<FillLike> {
public:
  FillLike(double value)
    : m_value(value)
  {
    transformation_(this, "fill")
      .input("a")
      .output("a")
      .types(Atypes::passAll)
      .func([](FillLike *obj, Args /*args*/, Rets rets) {
          rets[0].x.setConstant(obj->m_value);
        });
  }
protected:
  double m_value;
};

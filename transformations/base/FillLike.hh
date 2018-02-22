#pragma once

#include <string>

#include "GNAObject.hh"
#include "TypesFunctions.hh"

class FillLike: public GNASingleObject,
                public TransformationBind<FillLike> {
public:
  FillLike(double value)
    : m_value(value)
  {
    transformation_("fill")
      .input("a")
      .output("a")
      .types(TypesFunctions::passAll)
      .func([](FillLike *obj, Args /*args*/, Rets rets) {
          rets[0].x.setConstant(obj->m_value);
        });
  }
protected:
  double m_value;
};

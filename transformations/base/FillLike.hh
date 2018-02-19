#ifndef FILLLIKE_H
#define FILLLIKE_H

#include <string>

#include "GNAObject.hh"

class FillLike: public GNASingleObject,
                public TransformationBlock<FillLike> {
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

#endif // FILLLIKE_H

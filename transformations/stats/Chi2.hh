#ifndef CHI2_H
#define CHI2_H

#include "GNAObject.hh"
#include "Statistic.hh"

class Chi2: public GNASingleObject,
            public TransformationBlock<Chi2>,
            public Statistic {
public:
  Chi2() {
    transformation_(this, "chi2")
      .output("chi2")
      .types(&Chi2::checkTypes)
      .func(&Chi2::calculateChi2)
    ;
    m_transform = t_["chi2"];
  }

  void add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov);

  void checkTypes(Atypes args, Rtypes rets);
  void calculateChi2(Args args, Rets rets);

  virtual double value() override {
    return m_transform[0].x[0];
  }
protected:
  Handle m_transform;
};

#endif // CHI2_H

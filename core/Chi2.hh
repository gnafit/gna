#ifndef CHI2_H
#define CHI2_H

#include "GNAObject.hh"
#include "Statistic.hh"

class Chi2: public GNASingleObject,
            public Transformation<Chi2>,
            public Statistic {
public:
  Chi2() {
    transformation_(this, "chi2")
      .input("prediction")
      .input("data")
      .input("L")
      .output("chi2")
      .types([](Atypes args, Rtypes rets) {
          if (args[0].shape.size() != 1) {
            throw rets.error(rets[0]);
          }
          if (args[1].shape != args[0].shape) {
            throw rets.error(rets[0]);
          }
          if (args[2].shape.size() != 2 ||
              args[2].shape[0] != args[2].shape[1]) {
            throw rets.error(rets[0]);
          }
          if (args[2].shape[0] != args[0].shape[0]) {
            throw rets.error(rets[0]);
          }
          rets[0] = DataType().points().shape(1);
        })
      .func(&Chi2::calculateChi2)
    ;
    m_transform = t_["chi2"];
  }
  void calculateChi2(Args args, Rets rets);

  virtual double value() override {
    return m_transform[0].x[0];
  }
protected:
  Handle m_transform;
};

#endif // CHI2_H

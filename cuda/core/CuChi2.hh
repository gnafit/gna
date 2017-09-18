#ifndef CUCHI2_H
#define CUCHI2_H

#include "GNAObject.hh"
#include "Statistic.hh"

class CuChi2: public GNASingleObject,
            public Transformation<CuChi2>,
            public Statistic {
public:
  CuChi2() : InvMatCompFlag(false) {
    transformation_(this, "cuchi2")
      .output("cuchi2")
      .types(&CuChi2::checkTypes)
      .func(&CuChi2::calculateChi2)
    ;
    m_transform = t_["cuchi2"];
  }
  
  void add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov);

  void checkTypes(Atypes args, Rtypes rets);
  void calculateChi2(Args args, Rets rets);

  virtual double value() override {
    return m_transform[0].x[0];
  }
protected:
  bool InvMatCompFlag;
  std::vector<Eigen::MatrixXd> L;
  Handle m_transform;
};


#endif // CUCHI2_H

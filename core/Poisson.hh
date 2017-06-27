#ifndef POISSON_H
#define POISSON_H

#include "GNAObject.hh"
#include "Statistic.hh"


class Poisson: public GNASingleObject,
	       public Transformation<Poisson>,
	       public Statistic	 {
public:

 Poisson() {
    transformation_(this, "poisson")
      .output("poisson")
      .types(&Poisson::checkTypes)
      .func(&Poisson::calcPoisson)
    ;
    m_transform = t_["poisson"];
    approx = false;
  }
  bool approx;
  

  void add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov, bool ln_approx = false);

  void calcPoisson(Args args, Rets rets);
  void checkTypes(Atypes args, Rtypes rets);

  virtual double value() override {
    return m_transform[0].x[0];
  }

protected:
  Handle m_transform;
  
};

#endif

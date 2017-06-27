#ifndef POISSON_H
#define POISSON_H

#include "GNAObject.hh"
#include "Statistic.hh"


class Poisson: public GNASingleObject,
	       public Transformation<Poisson>,
	       public Statistic	 {
public:

 Poisson(bool ln_approx = false) {
    transformation_(this, "poisson")
        .output("poisson")
        .types(&Poisson::checkTypes)
        .func(ln_approx ? &Poisson::calcPoissonApprox : &Poisson::calcPoisson)
      ;
    m_transform = t_["poisson"];
  }
  
  

  void add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov);

  void calcPoissonApprox(Args args, Rets rets);

  void calcPoisson(Args args, Rets rets);

  void checkTypes(Atypes args, Rtypes rets);

  virtual double value() override {
    return m_transform[0].x[0];
  }

protected:
  Handle m_transform;
};

#endif

#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"


class Poisson: public GNASingleObject,
	       public TransformationBind<Poisson>,
	       public Statistic	 {
public:

 Poisson(bool ln_approx = false) {
    transformation_("poisson")
        .output("poisson")
        .types(&Poisson::checkTypes)
        .func(ln_approx ? &Poisson::calcPoissonApprox : &Poisson::calcPoisson)
      ;
    m_transform = t_["poisson"];
  }



  void add(SingleOutput &theory, SingleOutput &data);

  void calcPoissonApprox(FunctionArgs fargs);

  void calcPoisson(FunctionArgs fargs);

  void checkTypes(Atypes args, Rtypes rets);

  virtual double value() override {
    return m_transform[0].x[0];
  }

protected:
  Handle m_transform;
};

#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"

class Poisson: public GNASingleObject,
               public TransformationBind<Poisson>,
               public Statistic  {
public:
  Poisson(bool ln_approx=false);

  void add(SingleOutput &theory, SingleOutput &data);

  void calcPoissonApprox(FunctionArgs fargs);

  void calcPoisson(FunctionArgs fargs);

  void checkTypes(TypesFunctionArgs fargs);

  double value() override {
    return m_transform[0].x[0];
  }

protected:
  Handle m_transform;
};

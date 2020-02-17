#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"

class LogPoissonSplit: public GNAObject,
                      public TransformationBind<LogPoissonSplit>,
                      public Statistic  {
public:
  LogPoissonSplit(bool ln_approx=false);

  void add(SingleOutput &theory, SingleOutput &data);

  void calcPoissonConstApprox(FunctionArgs fargs);
  void calcPoissonConst(FunctionArgs fargs);
  void calcPoisson(FunctionArgs fargs);

  void checkTypesConst(TypesFunctionArgs fargs);
  void checkTypes(TypesFunctionArgs fargs);

  double value() override {
    return m_transform[0].x[0];
  }

protected:
  Handle m_transform;
};

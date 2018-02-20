#pragma once

#include "GNAObject.hh"
#include "UncertainParameter.hh"

class Derivative: public GNASingleObject,
                  public TransformationBind<Derivative> {
public:
  Derivative(Parameter<double> *x, double reldelta=1.0)
    : m_x(x), m_reldelta(reldelta)
  {
    transformation_(this, "derivative")
      .input("y")
      .output("dy")
      .types(Atypes::pass<0>)
      .func(&Derivative::calcDerivative)
    ;
  }

  void calcDerivative(Args args, Rets rets);
protected:
  Parameter<double> *m_x;
  double m_reldelta;
};

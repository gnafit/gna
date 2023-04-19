#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"

class Chi2CNPDiag: public GNASingleObject,
                   public TransformationBind<Chi2CNPDiag>,
                   public Statistic  {
public:
  Chi2CNPDiag();

  void add(SingleOutput &theory, SingleOutput &data);
  void calcChi2CNP(FunctionArgs fargs);
  void checkTypes(TypesFunctionArgs fargs);

  double value() override {
    return m_transform[0].x[0];
  }

protected:
  Handle m_transform;
};

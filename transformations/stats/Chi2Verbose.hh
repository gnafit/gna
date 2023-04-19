#pragma once

#include "GNAObject.hh"
#include "Statistic.hh"

class Chi2Verbose: public GNASingleObject,
            public TransformationBind<Chi2Verbose>,
            public Statistic {
public:
  Chi2Verbose() {
    transformation_("chi2")
      .output("chi2")
      .types(&Chi2Verbose::checkTypes)
      .func(&Chi2Verbose::calculateChi2Verbose)
    ;
    m_transform = t_["chi2"];
  }

  void add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov);

  void checkTypes(TypesFunctionArgs fargs);
  void calculateChi2Verbose(FunctionArgs fargs);

  double value() override {
    return m_transform[0].x[0];
  }
protected:
  Handle m_transform;
};

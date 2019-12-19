#pragma once

#include "Random.hh"
#include "GNAObject.hh"
#include "GNAObjectBind1N.hh"

class PoissonToyMC: public GNAObjectBind1N<double>,
                    public TransformationBind<PoissonToyMC> {
public:
  PoissonToyMC(bool autofreeze=true);

  void add(SingleOutput& theory, SingleOutput &cov) { add( theory ); }
  void add(SingleOutput& theory) { this->add_input(theory); }
  void nextSample();

  void reset() { m_distr.reset(); }

  TransformationDescriptor add_transformation(const std::string& name="");
protected:
  void calcTypes(TypesFunctionArgs fargs);
  void calcToyMC(FunctionArgs fargs);

  std::poisson_distribution<> m_distr;

  bool m_autofreeze;
};

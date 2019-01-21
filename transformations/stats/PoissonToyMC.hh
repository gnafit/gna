#pragma once

#include "Random.hh"
#include "GNAObject.hh"

class PoissonToyMC: public GNASingleObject,
                    public TransformationBind<PoissonToyMC> {
public:
  PoissonToyMC( bool autofreeze=true );

  void add(SingleOutput &theory, SingleOutput &cov) { add( theory ); }
  void add(SingleOutput &theory);
  void nextSample();

  void reset() { m_distr.reset(); }
protected:
  void calcTypes(TypesFunctionArgs fargs);
  void calcToyMC(FunctionArgs fargs);

  std::poisson_distribution<> m_distr;

  bool m_autofreeze;
};

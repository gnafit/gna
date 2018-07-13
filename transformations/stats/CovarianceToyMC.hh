#pragma once

#include "Random.hh"
#include "GNAObject.hh"

class CovarianceToyMC: public GNASingleObject,
                       public TransformationBind<CovarianceToyMC> {
public:
  CovarianceToyMC( bool autofreeze=true );

  void add(SingleOutput &theory, SingleOutput &cov);
  void nextSample();

  void reset() { m_distr.reset(); }
protected:
  void calcTypes(Atypes args, Rtypes rets);
  void calcToyMC(FunctionArgs fargs);

  std::normal_distribution<> m_distr;

  bool m_autofreeze;
};

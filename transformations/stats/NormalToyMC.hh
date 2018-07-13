#pragma once

#include "Random.hh"
#include "GNAObject.hh"

class NormalToyMC: public GNASingleObject,
                   public TransformationBind<NormalToyMC> {
public:
  NormalToyMC( bool autofreeze=true );

  void add(SingleOutput &theory, SingleOutput &sigma);

  void nextSample();

  void reset() { m_distr.reset(); }
protected:
  void calcTypes(Atypes args, Rtypes rets);
  void calcToyMC(FunctionArgs fargs);

  std::normal_distribution<> m_distr;

  bool m_autofreeze;
};

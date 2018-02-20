#ifndef NORMALTOYMC_H
#define NORMALTOYMC_H

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
  void calcToyMC(Args args, Rets rets);

  std::normal_distribution<> m_distr;

  bool m_autofreeze;
};

#endif // NORMALTOYMC_H

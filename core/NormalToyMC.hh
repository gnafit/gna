#ifndef NORMALTOYMC_H
#define NORMALTOYMC_H

#include "Random.hh"
#include "GNAObject.hh"

class NormalToyMC: public GNASingleObject,
                   public Transformation<NormalToyMC> {
public:
  NormalToyMC( bool autofreeze=true );

  void add(SingleOutput &theory, SingleOutput &sigma);

  void nextSample();
protected:
  void calcTypes(Atypes args, Rtypes rets);
  void calcToyMC(Args args, Rets rets);

  std::normal_distribution<> m_gen;

  bool m_autofreeze;
};

#endif // NORMALTOYMC_H

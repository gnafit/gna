#ifndef POISSONTOYMC_H
#define POISSONTOYMC_H

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
  void calcTypes(Atypes args, Rtypes rets);
  void calcToyMC(Args args, Rets rets);

  std::poisson_distribution<> m_distr;

  bool m_autofreeze;
};

#endif // POISSONTOYMC_H

#ifndef POISSONTOYMC_H
#define POISSONTOYMC_H

#include "Random.hh"
#include "GNAObject.hh"

class PoissonToyMC: public GNASingleObject,
                    public Transformation<PoissonToyMC> {
public:
  PoissonToyMC( bool autofreeze=true );

  void add(SingleOutput &theory, SingleOutput &cov);
  void nextSample();

protected:
  void calcTypes(Atypes args, Rtypes rets);
  void calcToyMC(Args args, Rets rets);

  std::poisson_distribution<> m_gen;

  bool m_autofreeze;
};

#endif // POISSONTOYMC_H

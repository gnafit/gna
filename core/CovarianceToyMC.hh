#ifndef COVARIANCETOYMC_H
#define COVARIANCETOYMC_H

#include "Random.hh"
#include "GNAObject.hh"

class CovarianceToyMC: public GNASingleObject,
                       public Transformation<CovarianceToyMC> {
public:
  CovarianceToyMC( bool autofreeze=true );

  void add(SingleOutput &theory, SingleOutput &cov);
  void nextSample();
  void seed(unsigned int s);
protected:
  void calcTypes(Atypes args, Rtypes rets);
  void calcToyMC(Args args, Rets rets);

  std::normal_distribution<> m_gen;

  bool m_autofreeze;
};

#endif // COVARIANCETOYMC_H

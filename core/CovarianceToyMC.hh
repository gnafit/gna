#ifndef COVARIANCETOYMC_H
#define COVARIANCETOYMC_H

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

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

  boost::mt19937 m_rand;
  boost::variate_generator<
    boost::mt19937&, boost::normal_distribution<>
  > m_gen{m_rand, boost::normal_distribution<>()};

  bool m_autofreeze;
};

#endif // COVARIANCETOYMC_H

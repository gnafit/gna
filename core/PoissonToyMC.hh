#ifndef POISSONTOYMC_H
#define POISSONTOYMC_H

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>

#include "GNAObject.hh"

class PoissonToyMC: public GNASingleObject,
                    public Transformation<PoissonToyMC> {
public:
  PoissonToyMC();

  void add(SingleOutput &theory, SingleOutput &cov);
  void nextSample();
  void seed(unsigned int s);
protected:
  void calcTypes(Atypes args, Rtypes rets);
  void calcToyMC(Args args, Rets rets);

  boost::mt19937 m_rand;
  boost::variate_generator<
    boost::mt19937&, boost::poisson_distribution<int>
  > m_gen{m_rand, boost::poisson_distribution<int>()};
};

#endif // POISSONTOYMC_H

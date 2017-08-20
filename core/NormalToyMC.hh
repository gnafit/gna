#ifndef NORMALTOYMC_H
#define NORMALTOYMC_H

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include "GNAObject.hh"

class NormalToyMC: public GNASingleObject,
                   public Transformation<NormalToyMC> {
public:
  NormalToyMC();

  void add(SingleOutput &theory, SingleOutput &sigma);

  void nextSample();
  void seed(unsigned int s);
protected:
  void calcTypes(Atypes args, Rtypes rets);
  void calcToyMC(Args args, Rets rets);

  boost::mt19937 m_rand;
  boost::variate_generator<
    boost::mt19937&, boost::normal_distribution<>
  > m_gen{m_rand, boost::normal_distribution<>()};
};

#endif // NORMALTOYMC_H

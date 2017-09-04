#ifndef NORMALTOYMC_H
#define NORMALTOYMC_H

#include "Random.hh"
#include <boost/random/normal_distribution.hpp>

#include "GNAObject.hh"

class NormalToyMC: public GNASingleObject,
                   public Transformation<NormalToyMC> {
public:
  NormalToyMC( bool autofreeze=true );

  void add(SingleOutput &theory, SingleOutput &sigma);

  void nextSample();
  void seed(unsigned int s);
protected:
  void calcTypes(Atypes args, Rtypes rets);
  void calcToyMC(Args args, Rets rets);

  boost::variate_generator<
    boost::mt19937&, boost::normal_distribution<>
  > m_gen{GNA::random_generator, boost::normal_distribution<>()};
  bool m_autofreeze;
};

#endif // NORMALTOYMC_H

#ifndef RANDOM_H
#define RANDOM_H 1

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

namespace GNA
{
  static boost::mt19937 random_generator;
} /* GNA */

#endif

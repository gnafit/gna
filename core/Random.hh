#ifndef RANDOM_H
#define RANDOM_H 1

#include <random>

namespace GNA
{
  namespace random
  {
      static std::random_device device;
      static std::mt19937       generator(device());
  } /* random */
} /* GNA */

#endif

#ifndef RANDOM_H
#define RANDOM_H 1

#include <random>

namespace GNA
{
  /*! \brief Common random generator
   *
   * Class holds random generator instance meant to be shared between random distributions
   */
  class Random
  {
  public:
    Random() : m_device(), m_generator( m_device() ) {};

    Random(Random&& other) = delete;
    Random(const Random& other) = delete;

    static std::mt19937* gen_ptr() { return &instance()->m_generator; }
    static std::mt19937& gen()     { return instance()->m_generator; }

    static void seed( unsigned long int seed ) { instance()->m_generator.seed( seed ); }

    static Random* instance() {
        static GNA::Random rnd_instance;
        return &rnd_instance;
    }

  private:
    std::random_device m_device;    /// Random device for generator initialization
    std::mt19937       m_generator; /// Mersenne-Twister random generator
  };

} /* GNA */

#endif

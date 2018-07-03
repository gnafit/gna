#pragma once

#include <functional>
#include <random>
#include <list>

namespace GNA
{
  /** \brief Common random generator
   *
   * Class holds random generator instance meant to be shared between random distributions
   * Random is ensured to be static and non-copiable
   */
  class Random
  {
  public:
    Random() : m_device(), m_generator( m_device() ) {};

    Random(Random&& other) = delete;
    Random(const Random& other) = delete;

    static std::mt19937* gen_ptr() { return &instance()->m_generator; }
    static std::mt19937& gen()     { return instance()->m_generator; }

    /** \brief Change the seed and call the callback functions */
    static void seed( unsigned long int seed ) {
        instance()->m_generator.seed( seed );
        for( auto& fcn : instance()->m_callbacks ){
            fcn();
        }
    }

    /** \brief Return statin instance */
    static Random* instance() {
        static GNA::Random rnd_instance;
        return &rnd_instance;
    }

    /** \brief Register a callback to be called after seed is changed */
    static void register_callback( std::function<void()> fcn ){ instance()->m_callbacks.push_back( fcn ); }

  private:
    std::random_device m_device;    /// Random device for generator initialization
    std::mt19937       m_generator; /// Mersenne-Twister random generator
    std::list<std::function<void()>> m_callbacks; /// Dependant distributions reset functions to be called after seed is set
  };

} /* GNA */

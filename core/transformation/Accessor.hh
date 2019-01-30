#pragma once

#include <string>

namespace TransformationTypes
{
  template<typename SourceFloatType, typename SinkFloatType> class BaseT;
  class Handle;
  /**
   * @brief Accessor gives an access to the Base's Entry instances by wrapping them into Handle.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class Accessor {
  public:
    using Base = BaseT<double,double>;

    Accessor() = default;                                 ///< Default constructor.
    Accessor(Base &parent): m_parent(&parent) { }         ///< Constructor. @param parent -- Base instance to access its Entry instances.
    Handle operator[](int idx) const;                     ///< Get a Handle for the i-th Entry.
    Handle operator[](const std::string &name) const;     ///< Get a Handle for the Entry by name.
    size_t size() const;                                  ///< Get number of Entry instances.
  private:
    Base *m_parent;                                       ///< Pointer to the Base that keeps Entry instances.
  }; /* class Accessor */

} /* TransformationTypes */


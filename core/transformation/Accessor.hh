#pragma once

#include <string>

namespace TransformationTypes
{
  template<typename SourceFloatType, typename SinkFloatType> class BaseT;

  using SourceFloatType=double;
  using SinkFloatType=double;

  template<typename SourceFloatType, typename SinkFloatType> class HandleT;
  /**
   * @brief Accessor gives an access to the Base's Entry instances by wrapping them into Handle.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class Accessor {
  public:
    using BaseType   = BaseT<SourceFloatType,SinkFloatType>;
    using HandleType = HandleT<SourceFloatType,SinkFloatType>;

    Accessor() = default;                                 ///< Default constructor.
    Accessor(BaseType &parent): m_parent(&parent) { }     ///< Constructor. @param parent -- Base instance to access its Entry instances.
    HandleType operator[](int idx) const;                 ///< Get a Handle for the i-th Entry.
    HandleType operator[](const std::string &name) const; ///< Get a Handle for the Entry by name.
    size_t size() const;                                  ///< Get number of Entry instances.
  private:
    BaseType *m_parent;                                   ///< Pointer to the Base that keeps Entry instances.
  }; /* class Accessor */

} /* TransformationTypes */


#ifndef OPENHANDLE_H
#define OPENHANDLE_H 1

#include "EntryHandle.hh"

namespace TransformationTypes
{
  /**
   * @brief User-end wrapper for the Entry class that gives user an access to the actual Entry.
   *
   * The class is used for the dependency tree plotting via graphviz module.
   *
   * @author Maxim Gonchar
   * @date 12.2017
   */
  class OpenHandle : public Handle {
  public:
      OpenHandle(const Handle& other) : Handle(other){}; ///< Constructor. @param other -- Handle instance.
      Entry* getEntry() { return m_entry; }              ///< Get the Entry pointer.
  };
} /* TransformationTypes */

#endif /* OPENHANDLE_H */

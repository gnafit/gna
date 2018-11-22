#pragma once

#include <string>

#include "Source.hh"
#include "OutputHandle.hh"

namespace TransformationTypes
{
  /**
   * @brief Source wrapper to make it user accessible from the Python.
   * @copydetails OutputHandle
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class InputHandle {
    friend class OutputHandle;
  public:
    /**
     * @brief Constructor.
     * @param s -- Source to access.
     */
    InputHandle(Source &source): m_source(&source) { }
    /**
     * @brief Clone constructor.
     * @param other -- other InputHandle instance to access its Source.
     */
    InputHandle(const InputHandle &other): InputHandle(*other.m_source) { }

    void connect(const OutputHandle &out) const; ///< Connect the Source to the other transformation's Sink via its OutputHandle

    const std::string &name() const { return m_source->name; }                 ///< Get Source's name.
    const std::string &label() const { return m_source->label; }               ///< Get Source's label.
    void  setLabel(const std::string& label) const { m_source->label=label; }  ///< Set Source's label.

    const void *rawptr() const { return static_cast<const void*>(m_source); } ///< Return Source's pointer as void pointer.
    const size_t hash() const { return reinterpret_cast<size_t>(rawptr()); }  ///< Return a Source's hash value based on it's pointer address.

    bool materialized() const { return m_source->materialized(); }            ///< Call Source::materialized(). @copydoc Source::materialized()

  protected:
    const OutputHandle output() const { return OutputHandle(*const_cast<Sink*>(m_source->sink)); }
    Source *m_source; ///< Pointer to the Source.
  }; /* class InputHandle */

} /* TransformationTypes */


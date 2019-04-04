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
  template<typename FloatType>
  class InputHandleT {
    template<typename FloatType1>
    friend class OutputHandleT;
  public:
    using SourceType = SourceT<FloatType>;
    /**
     * @brief Constructor.
     * @param s -- Source to access.
     */
    InputHandleT(SourceType &source): m_source(&source) { }
    /**
     * @brief Clone constructor.
     * @param other -- other InputHandleT instance to access its Source.
     */
    InputHandleT(const InputHandleT &other): InputHandleT(*other.m_source) { }

    void connect(const OutputHandleT<FloatType> &out) const;                            ///< Connect the Source to the other transformation's Sink via its OutputHandle
    void operator<<(const OutputHandleT<FloatType>& out) const { connect(out); }

    const std::string &name() const { return m_source->name; }                          ///< Get Source's name.
    const std::string &label() const { return m_source->attrs["_label"]; }              ///< Get Source's label.
    void  setLabel(const std::string& label) const { m_source->attrs["_label"]=label; } ///< Set Source's label.

    const void *rawptr() const { return static_cast<const void*>(m_source); }           ///< Return Source's pointer as void pointer.
    size_t hash() const { return reinterpret_cast<size_t>(rawptr()); }                  ///< Return a Source's hash value based on it's pointer address.

    bool materialized() const { return m_source->materialized(); }                      ///< Call Source::materialized(). @copydoc Source::materialized()
    bool bound() const { return m_source->sink!=nullptr; }                              ///< Return true if the source is bound to the sink.

    const OutputHandleT<FloatType> output() const { return OutputHandleT<FloatType>(*const_cast<SinkT<FloatType>*>(m_source->sink)); }
  protected:
    SourceType *m_source; ///< Pointer to the Source.
  }; /* class InputHandleT */

  template<typename FloatType>
  inline void operator>>(const OutputHandleT<FloatType>& output, const InputHandleT<FloatType>& input) { input.connect(output); }

  template void operator>><double>(const OutputHandleT<double>& output, const InputHandleT<double>& input);
#ifdef PROVIDE_SINGLE_PRECISION
  template void operator>><float>(const OutputHandleT<float>& output, const InputHandleT<float>& input);
#endif
} /* TransformationTypes */


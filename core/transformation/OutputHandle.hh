#pragma once

#include <string>

#include "Data.hh"
#include "TransformationEntry.hh"

namespace TransformationTypes
{
  /**
   * @brief Sink wrapper to make it user accessible from the Python.
   * InputHandle and OutputHandle classes give indirect access to Source and Sink instances
   * and enable users to connect them in a calculation chain.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename FloatType>
  class OutputHandleT {
    friend class InputHandle;
  public:
    /**
     * @brief Constructor.
     * @param s -- Sink to access.
     */
    OutputHandleT(SinkT<FloatType> &sink): m_sink(&sink) { }
    /**
     * @brief Clone constructor.
     * @param other -- other OutputHandleT instance to access its Sink.
     */
    OutputHandleT(const OutputHandleT &other): OutputHandleT(*other.m_sink) { }

    const std::string &name() const { return m_sink->name; }                   ///< Get Source's name.
    const std::string &label() const { return m_sink->label; }                 ///< Get Sink's label.
    void  setLabel(const std::string& label) const { m_sink->label=label; }    ///< Set Sink's label.

    bool check() const; ///< Check the Entry.
    void dump() const;  ///< Dump the Entry.

    const FloatType *data() const;                                              ///< Return pointer to the Sink's data buffer. Evaluate the data if needed in advance.
    const DataType &datatype() const { return m_sink->data->type; }             ///< Return Sink's DataType.

    const void *rawptr() const { return static_cast<const void*>(m_sink); }     ///< Return Source's pointer as void pointer.
    size_t hash() const { return reinterpret_cast<size_t>(rawptr()); }          ///< Return a Source's hash value based on it's pointer address.

    bool depends(changeable x) const;                                           ///< Check that Sink depends on a changeable.

  protected:
    const FloatType *view() const { return m_sink->data->x.data(); }            ///< Return pointer to the Sink's data buffer without evaluation.
    SinkT<FloatType> *m_sink;                                                   ///< Pointer to the Sink.
  }; /* class OutputHandleT */

  /**
   * @brief Return pointer to the Sink's data buffer. Evaluate the data if needed in advance.
   * @return pointer to the Sink's data buffer.
   */
  template<typename FloatType>
  inline const FloatType *OutputHandleT<FloatType>::data() const {
    m_sink->entry->touch();
    return this->view();
  }

  /**
   * @brief Check that Sink depends on a changeable.
   * Simply checks that Entry depends on a changeable.
   * @param x -- changeable to test.
   * @return true if depends.
   */
  template<typename FloatType>
  inline bool OutputHandleT<FloatType>::depends(changeable x) const {
    return m_sink->entry->tainted.depends(x);
  }

  /**
   * @brief Check the Entry.
   * @copydoc Entry::check()
   */
  template<typename FloatType>
  bool OutputHandleT<FloatType>::check() const {
    return m_sink->entry->check() && m_sink->data;
  }

  /**
   * @brief Dump the Entry.
   * @copydoc Entry::dump()
   */
  template<typename FloatType>
  void OutputHandleT<FloatType>::dump() const {
    m_sink->entry->dump(0);
  }

  using OutputHandle = OutputHandleT<double>;
} /* TransformationTypes */

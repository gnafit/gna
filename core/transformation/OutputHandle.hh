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
  class OutputHandle {
    friend class InputHandle;
  public:
    /**
     * @brief Constructor.
     * @param s -- Sink to access.
     */
    OutputHandle(Sink &sink): m_sink(&sink) { }
    /**
     * @brief Clone constructor.
     * @param other -- other OutputHandle instance to access its Sink.
     */
    OutputHandle(const OutputHandle &other): OutputHandle(*other.m_sink) { }

    const std::string &name() const { return m_sink->name; }                 ///< Get Source's name.

    bool check() const; ///< Check the Entry.
    void dump() const;  ///< Dump the Entry.

    const double *data() const;                                              ///< Return pointer to the Sink's data buffer. Evaluate the data if needed in advance.
    const DataType &datatype() const { return m_sink->data->type; }          ///< Return Sink's DataType.

    const void *rawptr() const { return static_cast<const void*>(m_sink); }  ///< Return Source's pointer as void pointer.
    const size_t hash() const { return reinterpret_cast<size_t>(rawptr()); } ///< Return a Source's hash value based on it's pointer address.

    bool depends(changeable x) const;                                        ///< Check that Sink depends on a changeable.

  private:
    const double *view() const { return m_sink->data->x.data(); }            ///< Return pointer to the Sink's data buffer without evaluation.
    Sink *m_sink;                                                            ///< Pointer to the Sink.
  }; /* class OutputHandle */

  /**
   * @brief Return pointer to the Sink's data buffer. Evaluate the data if needed in advance.
   * @return pointer to the Sink's data buffer.
   */
  inline const double *OutputHandle::data() const {
    m_sink->entry->touch();
    return this->view();
  }

  /**
   * @brief Check that Sink depends on a changeable.
   * Simply checks that Entry depends on a changeable.
   * @param x -- changeable to test.
   * @return true if depends.
   */
  inline bool OutputHandle::depends(changeable x) const {
    return m_sink->entry->tainted.depends(x);
  }

} /* TransformationTypes */

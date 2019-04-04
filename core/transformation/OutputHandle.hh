#pragma once

#include <string>

#include "changeable.hh"
#include "Sink.hh"
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
    template<typename FloatType1>
    friend class InputHandleT;
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

    const std::string &name() const { return m_sink->name; }                          ///< Get Source's name.
    const std::string &label() const { return m_sink->attrs["_label"]; }              ///< Get Sink's label.
    void  setLabel(const std::string& label) const { m_sink->attrs["_label"]=label; } ///< Set Sink's label.

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
} /* TransformationTypes */

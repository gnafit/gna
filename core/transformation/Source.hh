#pragma once

#include <boost/noncopyable.hpp>
#include <string>
#include <iostream>
#include "fmt/format.h"
#include <utility>
using fmt::format;

#include "Sink.hh"
#include "TransformationDebug.hh"

namespace TransformationTypes
{
  /**
   * @brief Definition of a single transformation input (Source).
   *
   * Source instance is a link to the other transformation Entry's Sink,
   * that carries the transformation output.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename FloatType>
  struct SourceT: public boost::noncopyable {
    using EntryType  = EntryT<FloatType,FloatType>;
    using SourceType = SourceT<FloatType>;
    using SinkType   = SinkT<FloatType>;
    using DataType   = Data<FloatType>;
    /**
     * @brief Constructor.
     * @param name -- Source name.
     * @param entry -- Entry pointer Source belongs to.
     * @param inactive -- if true, source becomes inactive and will not be subscribed to other taintflags.
     */
    SourceT(std::string name, EntryType *entry, bool inactive=false)
      : name(std::move(name)), entry(entry), inactive(inactive) { }
    /**
     * @brief Clone constructor.
     * @param name -- other Source to get the name from.
     * @param entry -- Entry pointer Source belongs to.
     */
    SourceT(const SourceType &other, EntryType *entry)
      : name(other.name), entry(entry), inactive(other.inactive), attrs(other.attrs) { }

    void connect(SinkType *newsink);      ///< Connect the Source to the Sink.

    /**
     * @brief Check if the input data is allocated.
     * @return true if input data is allocated.
     */
    bool materialized() const {
      return sink && sink->materialized();
    }

    const DataType* getData() const {return sink ? sink->getData() : nullptr;}

    size_t hash() const { return reinterpret_cast<size_t>((void*)this); } ///< Return source address as size_t

    #ifdef GNA_CUDA_SUPPORT
    void requireGPU() const { if(sink) sink->requireGPU(); }
    #endif

    std::string name;                             ///< Source's name.
    const SinkType *sink = nullptr;               ///< Pointer to the Sink the Source is connected to.
    EntryType *entry;                             ///< Entry pointer the Source belongs to.
    bool inactive=false;                          ///< Source is inactive (taintflag will not be subscribed)
    Attrs attrs;                                  ///< Map with source attributes
  };
} /* namespace TransformationTypes */


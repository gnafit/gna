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
  struct Source: public boost::noncopyable {
    /**
     * @brief Constructor.
     * @param name -- Source name.
     * @param entry -- Entry pointer Source belongs to.
     * @param inactive -- if true, source becomes inactive and will not be subscribed to other taintflags.
     */
    Source(std::string name, Entry *entry, bool inactive=false)
      : name(std::move(name)), entry(entry), inactive(inactive) { }
    /**
     * @brief Clone constructor.
     * @param name -- other Source to get the name from.
     * @param entry -- Entry pointer Source belongs to.
     */
    Source(const Source &other, Entry *entry)
      : name(other.name), label(other.label), entry(entry), inactive(other.inactive) { }

    void connect(Sink *newsink);                   ///< Connect the Source to the Sink.

    /**
     * @brief Check if the input data is allocated.
     * @return true if input data is allocated.
     */
    bool materialized() const {
      return sink && sink->data;
    }

    const Data<double>* getData() const {return sink->getData();}

    std::string name;                             ///< Source's name.
    std::string label;                            ///< Source's label.
    const Sink *sink = nullptr;                   ///< Pointer to the Sink the Source is connected to.
    Entry *entry;                                 ///< Entry pointer the Source belongs to.
    bool inactive=false;                          ///< Source is inactive (taintflag will not be subscribed)
  };
} /* namespace TransformationTypes */


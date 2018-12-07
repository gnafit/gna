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
     */
    Source(std::string name, Entry *entry)
      : name(std::move(name)), entry(entry) { }
    /**
     * @brief Clone constructor.
     * @param name -- other Source to get the name from.
     * @param entry -- Entry pointer Source belongs to.
     */
    Source(const Source &other, Entry *entry)
      : name(other.name), label(other.label), entry(entry) { }

    void connect(Sink *newsink);                   ///< Connect the Source to the Sink.

    /**
     * @brief Check if the input data is allocated.
     * @return true if input data is allocated.
     */
    bool materialized() const {
      return sink && sink->data;
    }
    std::string name;                             ///< Source's name.
    std::string label;                            ///< Source's label.
    const Sink *sink = nullptr;                   ///< Pointer to the Sink the Source is connected to.
    Entry *entry;                                 ///< Entry pointer the Source belongs to.
  };
} /* namespace TransformationTypes */


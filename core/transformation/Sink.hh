#pragma once

#include <boost/noncopyable.hpp>
#include <string>
#include <memory>
#include <utility>

#include "Data.hh"

namespace TransformationTypes
{
  struct Entry;
  struct Source;

  /**
   * @brief Definition of a single transformation output (Sink).
   *
   * Sink instance carries the actual Data.
   *
   * It also knows where this data is connected to (Sink::sources).
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename FloatType>
  struct SinkT: public boost::noncopyable {
    using DataType = Data<FloatType>;

    /**
     * @brief Constructor.
     * @param name -- Sink name.
     * @param entry -- Entry pointer Sink belongs to.
     */
    SinkT(std::string name, Entry *entry)
      : name(std::move(name)), entry(entry) { }
    /**
     * @brief Clone constructor.
     * @param name -- other Sink to get the name from.
     * @param entry -- Entry pointer Sink belongs to.
     */
    SinkT(const SinkT<FloatType> &other, Entry *entry)
      : name(other.name), label(other.label), entry(entry) { }

    DataType* getData() {return data.get();}
    const DataType* getData() const {return const_cast<const DataType*>(data.get());}

    std::string name;                    ///< Sink's name.
    std::string label;                   ///< Sink's label.
    std::unique_ptr<DataType> data;      ///< Sink's Data.
    std::vector<Source*> sources;        ///< Container with Source pointers which use this Sink as their input.
    Entry *entry;                        ///< Pointer to the transformation Entry this Sink belongs to.
  };

  using Sink = SinkT<double>;
} /* TransformationTypes */

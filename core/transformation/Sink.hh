#pragma once

#include <boost/noncopyable.hpp>
#include <string>
#include <memory>
#include <utility>

#include "Data.hh"

namespace TransformationTypes
{
  template<typename SourceFloatType,typename SinkFloatType> struct EntryT;

  template<typename FloatType> struct SourceT;

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
    using EntryType = EntryT<FloatType,FloatType>;
    using SourceType = SourceT<FloatType>;
    using SinkType = SinkT<FloatType>;
    using DataType = Data<FloatType>;
    using DataPtr  = std::unique_ptr<DataType>;

    /**
     * @brief Constructor.
     * @param name -- Sink name.
     * @param entry -- Entry pointer Sink belongs to.
     */
    SinkT(std::string name, EntryType *entry)
      : name(std::move(name)), entry(entry) { }
    /**
     * @brief Clone constructor.
     * @param name -- other Sink to get the name from.
     * @param entry -- Entry pointer Sink belongs to.
     */
    SinkT(const SinkType &other, EntryType *entry)
      : name(other.name), label(other.label), entry(entry) { }

    bool materialized() const { return (bool)data; } ///< Check if data is initialized

    DataType* getData() {return data.get();}
    const DataType* getData() const {return const_cast<const DataType*>(data.get());}

    std::string name;                    ///< Sink's name.
    std::string label;                   ///< Sink's label.
    DataPtr data;                        ///< Sink's Data.
    std::vector<SourceType*> sources;    ///< Container with Source pointers which use this Sink as their input.
    EntryType *entry;                    ///< Pointer to the transformation Entry this Sink belongs to.
  };
} /* TransformationTypes */

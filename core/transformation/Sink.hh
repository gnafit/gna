#pragma once

#include <boost/noncopyable.hpp>
#include <string>
#include <map>
#include <memory>
#include <utility>

#include "Data.hh"

namespace TransformationTypes
{
  using Attrs = std::map<std::string,std::string>;
  template<typename SourceFloatType,typename SinkFloatType> class EntryT;

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
      : name(other.name), entry(entry), attrs(other.attrs) {  }

    bool materialized() const { return (bool)data; } ///< Check if data is initialized

    DataType* getData() {return data.get();}
    const DataType* getData() const {return const_cast<const DataType*>(data.get());}

    size_t hash() const { return reinterpret_cast<size_t>((void*)this); } ///< Return sink address as size_t

    std::string name;                    ///< Sink's name.
    DataPtr data;                        ///< Sink's Data.
    std::vector<SourceType*> sources;    ///< Container with Source pointers which use this Sink as their input.
    EntryType *entry;                    ///< Pointer to the transformation Entry this Sink belongs to.
    Attrs attrs;                         ///< Map with sink attributes
  };
} /* TransformationTypes */

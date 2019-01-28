#pragma once

#include <boost/noncopyable.hpp>
#include <string>
#include <memory>

#include "Data.hh"

namespace TransformationTypes
{
  template<typename SourceType,typename SinkType> struct EntryT;
  using Entry = EntryT<double,double>;

  /**
   * @brief Definition of a transformation internal data Storage.
   *
   * @author Maxim Gonchar
   * @date 17.07.2018
   */
  template<typename FloatType>
  struct StorageT: public boost::noncopyable {
    using DataType = Data<FloatType>;
    /**
     * @brief Constructor.
     * @param entry -- Entry pointer Storage belongs to.
     * @exception std::runtime_error in case entry==nullptr
     */
    StorageT(Entry *entry=nullptr) : entry(entry) { if(!entry) throw std::runtime_error("Storage initialized without entry"); }
    /**
     * @brief Clone constructor.
     * @param entry -- Entry pointer Storage belongs to.
     */
    StorageT(const StorageT<FloatType> &other, Entry *entry)
      : entry(entry) { }

    DataType*       getData()       {return data.get();}
    const DataType* getData() const {return data.get();}

    bool materialized() const { return (bool)data; } ///< Check if data is initialized

    std::string name;                    ///< Storage's name.
    std::string label;                   ///< Storage's label.
    std::unique_ptr<DataType> data;      ///< Storage's Data.
    Entry *entry;                        ///< Pointer to the transformation Entry this Storage belongs to.
  };

  using Storage = StorageT<double>;
} /* TransformationTypes */

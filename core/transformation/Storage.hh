#pragma once

#include <boost/noncopyable.hpp>
#include <string>
#include <memory>

#include "Data.hh"

namespace TransformationTypes
{
  struct Entry;

  /**
   * @brief Definition of a transformation internal data Storage.
   *
   * @author Maxim Gonchar
   * @date 17.07.2018
   */
  struct Storage: public boost::noncopyable {
    /**
     * @brief Constructor.
     * @param entry -- Entry pointer Storage belongs to.
     */
    Storage(Entry *entry) : entry(entry) { }
    /**
     * @brief Clone constructor.
     * @param entry -- Entry pointer Storage belongs to.
     */
    Storage(const Storage &other, Entry *entry)
      : entry(entry) { }

    std::string name;                           ///< Storage's name.
    std::string label;                          ///< Storage's label.
    std::unique_ptr<Data<double>> data;         ///< Storage's Data.
    Entry *entry;                               ///< Pointer to the transformation Entry this Storage belongs to.
  };
} /* TransformationTypes */

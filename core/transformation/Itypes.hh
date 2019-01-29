#pragma once

#include <string>
#include <memory>

#include "Data.hh"
#include "TransformationEntry.hh"
#include "TransformationErrors.hh"

namespace TransformationTypes
{
  /**
   * @brief Acess the transformation Storage data types.
   *
   * @author Maxim Gonchar
   * @date 17.07.2018
   */
  template<typename SourceFloatType, typename SinkFloatType>
  struct ItypesT {
  public:
    using EntryImpl   = EntryT<SourceFloatType,SinkFloatType>;
    using StorageImpl = StorageT<SourceFloatType>;

    /**
     * @brief Itypes constructor.
     *
     * @param e -- Entry instance.
     */
    ItypesT(const EntryImpl *e)
      : m_entry(e), m_types(new std::vector<DataType>())
      { }

    /**
     * @brief Get i-th Storage DataType.
     * @param i -- Storage index.
     * @return i-th Storage DataType.
     */
    DataType &operator[](int i);

    /**
     * @brief Get number of Storage instances.
     * @return number of storages.
     */
    size_t size() const { return m_types->size(); }

    /**
     * @brief Storage type exception.
     * @param dt -- incorrect DataType.
     * @param message -- exception message.
     * @return exception.
     */
    StorageTypeError error(const DataType &dt, const std::string &message = "");

    /**
     * @brief Get Entry's name
     * @return Entry's name
     */
    const std::string &name() const { return m_entry->name; }

  protected:
    const EntryImpl *m_entry; ///< Entry instance.
    std::shared_ptr<std::vector<DataType> > m_types; ///< the Storage data types.
  }; /* struct Itypes */
} /* TransformationTypes */


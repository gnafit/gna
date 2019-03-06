#pragma once

#include "Data.hh"
#include "TransformationErrors.hh"

namespace TransformationTypes
{
  template<typename SourceFloatType, typename SinkFloatType> struct EntryT;
  /**
   * @brief Access the transformation Storage instances.
   *
   * Ints instance is passed to the Entry::fun function and is used to store internal data of the transformation.
   *
   * @author Maxim Gonchar
   * @date 18.07.2018
   */
  template<typename SourceFloatType, typename SinkFloatType>
  struct IntsT {
  public:
    using EntryType = EntryT<SourceFloatType,SinkFloatType>;
    using DataType  = Data<SourceFloatType>;
    /**
     * @brief Ints constructor.
     * @param e -- Entry instance. Ints will get access to Entry's storages.
     */
    IntsT(EntryType *e): m_entry(e) { }

    /**
     * @brief Get i-th Storage Data.
     * @param i -- index of a Storage.
     * @return i-th Storage's Data.
     */
    DataType &operator[](int i) const;

    /**
     * @brief Get number of transformation storages.
     * @return Number of transformation storage instances.
     */
    size_t size() const;

    /**
     * @brief Calculation error exception.
     * @param message -- exception message.
     * @return exception.
     */
    CalculationError<EntryType> error(const std::string &message = "") const;

  private:
    EntryType *m_entry; ///< Entry instance to access Storage instances.
  };
} /* TransformationTypes */

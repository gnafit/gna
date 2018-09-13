#pragma once

#include "Data.hh"
#include "TransformationEntry.hh"
#include "TransformationErrors.hh"

namespace TransformationTypes
{
  /**
   * @brief Access the transformation Storage instances.
   *
   * Ints instance is passed to the Entry::fun function and is used to store internal data of the transformation.
   *
   * @author Maxim Gonchar
   * @date 18.07.2018
   */
  struct Ints {
  public:
    /**
     * @brief Ints constructor.
     * @param e -- Entry instance. Ints will get access to Entry's storages.
     */
    Ints(Entry *e): m_entry(e) { }

    /**
     * @brief Get i-th Storage Data.
     * @param i -- index of a Storage.
     * @return i-th Storage's Data.
     */
    Data<double> &operator[](int i) const;

    /**
     * @brief Get number of transformation storages.
     * @return Number of transformation storage instances.
     */
    size_t size() const { return m_entry->storages.size(); }

    /**
     * @brief Calculation error exception.
     * @param message -- exception message.
     * @return exception.
     */
    CalculationError error(const std::string &message = "") const;

  private:
    Entry *m_entry; ///< Entry instance to access Storage instances.
  };
} /* TransformationTypes */

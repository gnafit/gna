#pragma once

#include "Data.hh"
#include "TransformationEntry.hh"

namespace TransformationTypes
{
  /**
   * @brief Access the transformation inputs.
   *
   * Args instance is passed to the Entry::fun function and is used to retrieve input data for the transformation.
   *
   * Args gives read-only access to the Source instances through Entry instance.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Args {

  public:
    /**
     * @brief Args constructor.
     * @param e -- Entry instance. Args will get access to Entry's sources.
     */
    Args(const Entry *e): m_entry(e) { }

    /**
     * @brief Get i-th Source Data.
     * @param i -- index of a Source.
     * @return i-th Sources's Data as input (const).
     */
    const Data<double> &operator[](int i) const;

    /**
     * @brief Get number of transformation sources.
     * @return Number of transformation Source instances.
     */
    size_t size() const { return m_entry->sources.size(); }
  private:
    const Entry *m_entry; ///< Entry instance to access Sources.
  }; /* struct Args */
} /* TransformationTypes */

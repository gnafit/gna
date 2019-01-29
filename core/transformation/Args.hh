#pragma once

#include "Data.hh"

namespace TransformationTypes
{
  template<typename SourceFloatType, typename SinkFloatType> struct EntryT;
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
  template <typename SourceFloatType, typename SinkFloatType>
  struct ArgsT {
  public:
    using EntryImpl = EntryT<SourceFloatType,SinkFloatType>;
    using DataImpl  = Data<SourceFloatType>;
    /**
     * @brief Args constructor.
     * @param e -- Entry instance. Args will get access to Entry's sources.
     */
    ArgsT(const EntryImpl *e): m_entry(e) { }

    /**
     * @brief Get i-th Source Data.
     * @param i -- index of a Source.
     * @return i-th Sources's Data as input (const).
     */
    const DataImpl &operator[](int i) const;

    void touch() const;  ///< Touch all the sources.
    size_t size() const; ///< Get number of transformation sources.
  private:
    const EntryImpl *m_entry; ///< Entry instance to access Sources.
  }; /* struct Args */
} /* TransformationTypes */

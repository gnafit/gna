#pragma once

#include <string>
#include <memory>

#include "Data.hh"
#include "TransformationEntry.hh"
#include "TransformationErrors.hh"
#include "Sink.hh"

namespace TransformationTypes
{
  /**
   * @brief Storage for the new transformation's outputs' DataType types.
   *
   * It's needed to store the derived outputs' DataType types.
   *
   * Rtypes instance is passed to each of the Entry's TypeFunction objects.
   *
   * @note Rtypes will NOT write to Entry's output DataType types by itself. The actual assignment happens in the Entry::evaluateTypes() method.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename SourceFloatType, typename SinkFloatType>
  struct RtypesT {
  public:
    using EntryImpl = EntryT<SourceFloatType,SinkFloatType>;
    using DataImpl  = Data<SourceFloatType>;
    using SinkImpl  = SinkT<SourceFloatType>;
    using ErrorType = SinkTypeError<SinkImpl>;
    /**
     * @brief Rtypes constructor.
     *
     * Rtypes will NOT write to Entry's output DataType types by itself.
     *
     * @param e -- Entry instance.
     */
    RtypesT(const EntryImpl *e)
      : m_entry(e), m_types(new std::vector<DataType>(e->sinks.size()))
      { }

    /**
     * @brief Get i-th Sink DataType.
     * @param i -- Sink index.
     * @return i-th Sink DataType.
     */
    DataType &operator[](int i);

    /**
     * @brief Get i-th Sink DataType.
     * @param i -- Sink index.
     * @return i-th Sink DataType.
     */
    const DataType &operator[](int i) const;

    /**
     * @brief Get number of Sink instances.
     * @return number of sinks.
     */
    size_t size() const { return m_types->size(); }

    /**
     * @brief Sink type exception.
     * @param dt -- incorrect DataType.
     * @param message -- exception message.
     * @return exception.
     */
    ErrorType error(const DataType &dt, const std::string &message = "");

    /**
     * @brief Get Entry's name
     * @return Entry's name
     */
    const std::string &name() const { return m_entry->name; }

  protected:
    const EntryImpl *m_entry; ///< Entry instance.

    std::shared_ptr<std::vector<DataType> > m_types; ///< Storage for the output DataType types.
  }; /* struct Rtypes */
} /* TransformationTypes */

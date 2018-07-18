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
  struct Rtypes {
  public:
    /**
     * @brief Rtypes constructor.
     *
     * Rtypes will NOT write to Entry's output DataType types by itself.
     *
     * @param e -- Entry instance.
     */
    Rtypes(const Entry *e)
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
    SinkTypeError error(const DataType &dt, const std::string &message = "");

    /**
     * @brief Get Entry's name
     * @return Entry's name
     */
    const std::string &name() const { return m_entry->name; }

  protected:
    const Entry *m_entry; ///< Entry instance.
    std::shared_ptr<std::vector<DataType> > m_types; ///< Storage for the output DataType types.
  }; /* struct Rtypes */
} /* TransformationTypes */

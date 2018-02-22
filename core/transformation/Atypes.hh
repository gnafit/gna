#pragma once

#include <string>

#include "TransformationEntry.hh"
#include "TransformationErrors.hh"
#include "Rtypes.hh"

namespace TransformationTypes
{
  /**
   * @brief Access the transformation inputs' DataType (read only).
   *
   * It's needed to:
   *   - check the consistency of the inputs in the run time.
   *   - derive the output DataType instances.
   *
   * Atypes instance is passed to each of the Entry's TypeFunction objects.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Atypes {
    /**
     * @brief An exception for uninitialized Source instance
     */
    class Undefined {
    public:
      Undefined(const Source *s = nullptr) : source(s) { }
      const Source *source;
    };
    /**
     * @brief Atypes constructor.
     * @param e -- Entry instance. Atypes will get access to Entry's source types.
     */
    Atypes(const Entry *e): m_entry(e) { }

    /**
     * @brief Direct access to Sink instance, which is used as Source for the transformation.
     *
     * @param i -- Source number to return its Sink.
     * @return i-th Source's Sink instance.
     *
     * @exception Undefined in case input data is not initialized.
     */
    const Sink *sink(int i) const {
      if (!m_entry->sources[i].materialized()) {
        throw Undefined(&m_entry->sources[i]);
      }
      return m_entry->sources[i].sink;
    }

    /**
     * @brief Get i-th Source DataType (const).
     * @param i -- Source index.
     * @return i-th Source DataType.
     */
    const DataType &operator[](int i) const {
      return sink(i)->data->type;
    }

    /**
     * @brief Get number of Source instances.
     * @return number of sources.
     */
    size_t size() const { return m_entry->sources.size(); }

    /**
     * @brief Source type exception.
     * @param dt -- incorrect DataType.
     * @param message -- exception message.
     * @return exception.
     */
    SourceTypeError error(const DataType &dt, const std::string &message = "");

    /**
     * @brief Get Entry's name
     * @return Entry's name
     */
    const std::string &name() const { return m_entry->name; }

    /**
     * @brief Empty Undefined exception.
     * @return Empty Undefined exception.
     */
    Undefined undefined() { return Undefined(); }
  private:
    const Entry *m_entry; ///< Entry instance to access Source DataType.
  };
} /* TransformationBase */

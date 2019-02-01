#pragma once

#include <string>

#include "TransformationEntry.hh"
#include "TransformationErrors.hh"

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
  template<typename SourceFloatType, typename SinkFloatType>
  struct AtypesT {
    using EntryImpl  = EntryT<SourceFloatType,SinkFloatType>;
    using SourceImpl = SourceT<SourceFloatType>;
    using SinkImpl   = SinkT<SourceFloatType>;

    /**
     * @brief An exception for uninitialized Source instance
     */
    class Undefined {
    public:
      Undefined(const SourceImpl *s = nullptr) : source(s) { }
      const SourceImpl *source;
    };
    /**
     * @brief Atypes constructor.
     * @param e -- Entry instance. Atypes will get access to Entry's source types.
     */
    AtypesT(const EntryImpl *e): m_entry(e) { }

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
    SourceTypeError<SourceImpl> error(const DataType &dt, const std::string &message = "");

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

  protected:
    /**
     * @brief Direct access to Sink instance, which is used as Source for the transformation.
     *
     * @param i -- Source number to return its Sink.
     * @return i-th Source's Sink instance.
     *
     * @exception Undefined in case input data is not initialized.
     */
    const SinkImpl *sink(int i) const {
      if (!m_entry->sources[i].materialized()) {
        throw Undefined(&m_entry->sources[i]);
      }
      return m_entry->sources[i].sink;
    }

  private:
    const EntryImpl *m_entry; ///< Entry instance to access Source DataType.
  };
} /* TransformationBase */

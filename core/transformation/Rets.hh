#pragma once

#include "Data.hh"
#include "TransformationErrors.hh"

namespace TransformationTypes
{
  template<typename SourceFloatType, typename SinkFloatType> class EntryT;

  /**
   * @brief Access the transformation outputs.
   *
   * Rets instance is passed to the Entry::fun function and is used to write output data of the transformation.
   *
   * Rets gives write access to the Sink instances through Entry instance.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename SourceFloatType, typename SinkFloatType>
  struct RetsT {
  public:
    using EntryType = EntryT<SourceFloatType,SinkFloatType>;
    using DataType  = Data<SinkFloatType>;
    /**
     * @brief Rets constructor.
     * @param e -- Entry instance. Rets will get access to Entry's sinks.
     */
    RetsT(EntryType *e): m_entry(e) { }

    /**
     * @brief Get i-th Sink Data.
     * @param i -- index of a Sink.
     * @return i-th Sink's Data as output.
     */
    DataType &operator[](int i) const;

    /**
     * @brief Get number of transformation sinks.
     * @return Number of transformation Sink instances.
     */
    size_t size() const;

    /**
     * @brief Calculation error exception.
     * @param message -- exception message.
     * @return exception.
     */
    CalculationError<EntryType> error(const std::string &message = "") const;

    /**
     * @brief Freeze the Entry.
     *
     * While entry is frozen the taintflag is not propagated. Entry is always up to date.
     */
    void freeze();

    /**
     * @brief Untaint the Entry.
     *
     * Set Entry's taintflag to false.
     */
    void untaint();

    /**
     * @brief Unfreeze the Entry.
     *
     * Enables the taintflag propagation.
     */
    void unfreeze();

  private:
    EntryType *m_entry; ///< Entry instance to access Sinks.
  };
} /* TransformationTypes */

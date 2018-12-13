#pragma once

#include "Data.hh"
#include "TransformationEntry.hh"
#include "TransformationErrors.hh"

namespace TransformationTypes
{
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
  struct Rets {
  public:
    /**
     * @brief Rets constructor.
     * @param e -- Entry instance. Rets will get access to Entry's sinks.
     */
    Rets(Entry *e): m_entry(e) { }

    /**
     * @brief Get i-th Sink Data.
     * @param i -- index of a Sink.
     * @return i-th Sink's Data as output.
     */
    Data<double> &operator[](int i) const;

    /**
     * @brief Get number of transformation sinks.
     * @return Number of transformation Sink instances.
     */
    size_t size() const { return m_entry->sinks.size(); }

    /**
     * @brief Calculation error exception.
     * @param message -- exception message.
     * @return exception.
     */
    CalculationError error(const std::string &message = "") const;

    /**
     * @brief Freeze the Entry.
     *
     * While entry is frozen the taintflag is not propagated. Entry is always up to date.
     */
    void freeze()  { m_entry->tainted.freeze(); }

    /**
     * @brief Untaint the Entry.
     *
     * Set Entry's taintflag to false.
     */
    void untaint()  { m_entry->tainted=false; }

    /**
     * @brief Unfreeze the Entry.
     *
     * Enables the taintflag propagation.
     */
    void unfreeze()  { m_entry->tainted.unfreeze(); }

  private:
    Entry *m_entry; ///< Entry instance to access Sinks.
  };
} /* TransformationTypes */

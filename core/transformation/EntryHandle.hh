#pragma once

#include <string>
#include <vector>

#include "Data.hh"
#include "InputHandle.hh"
#include "OutputHandle.hh"

class SingleOutput;
namespace TransformationTypes
{
  /**
   * @brief User-end Entry wrapper.
   *
   * This class gives an access to the transformation Entry.
   * It is inherited by TransformationDescriptor.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class Handle {
  public:
    Handle(): m_entry(nullptr) { }                            ///< Default constructor.
    Handle(Entry &entry) : m_entry(&entry) { }                ///< Constructor. @param entry -- an Entry instance to wrap.
    Handle(const Handle &other): Handle(*other.m_entry) { }   ///< Constructor. @param other -- Handle instance to get Entry to wrap.

    const std::string &name() const { return m_entry->name; }                 ///< Get entry name.
    const std::string &label() const { return m_entry->label; }               ///< Get entry label.
    void  setLabel(const std::string& label) const { m_entry->label=label; }  ///< Set entry label.
    std::vector<InputHandle> inputs() const;                                  ///< Get vector of inputs.
    std::vector<OutputHandle> outputs() const;                                ///< Get vector of outputs.

    /**
     * @brief Add named input.
     * @param name -- Source name.
     * @return InputHandle for the newly created Source.
     */
    InputHandle input(const std::string &name) {
      return m_entry->addSource(name);
    }

    InputHandle input(SingleOutput &output);                 ///< Create a new input and connect to the SingleOutput transformation.

    /**
     * @brief Add new named output.
     *
     * @param name -- new Sink's name.
     * @return OutputHandle for the new Sink.
     */
    OutputHandle output(const std::string &name) {
      return m_entry->addSink(name);
    }
    OutputHandle output(SingleOutput &output);               ///< Create a new output with a same name as SingleOutput's output.

    /**
     * @brief Return i-th Entry's Sink's data.
     *
     * The transformation function is evaluated if needed.
     *
     * @param i -- index of an output.
     * @return Data instance.
     */
    const Data<double> &operator[](int i) const { return m_entry->data(i); }

    /**
     * @brief Trigger an update of an Entry by simulating access to the i-th data.
     *
     * @param i -- Entry's Sink's index.
     */
    void update(int i) const { (void)m_entry->data(i); }
    void updateTypes() { m_entry->updateTypes(); }          ///< Call Entry::evaluateTypes(). @copydoc Entry::evaluateTypes()

    void unfreeze() { m_entry->tainted.unfreeze(); }        ///< Unfreeze Entry's taintflag.

    void taint() { m_entry->tainted.taint(); }              ///< Taint the Entry's taintflag. The outputs will be evaluated upon request.
    bool tainted() { return m_entry->tainted; }             ///< Return the Entry's taintflag status.

    /**
     * @brief Switch the active Function.
     *
     * @copydoc Entry::switchFunction()
     */
    void switchFunction(const std::string& name) {
      m_entry->switchFunction(name);
    }

    bool check() const { return m_entry->check(); }         ///< Call Entry::check(). @copydoc Entry::check()
    void dump() const { m_entry->dump(0); }                 ///< Call Entry::dump(). @copydoc Entry::dump()
    void dumpObj() const;                                   ///< Print Entry's Sink and Source instances and their connection status.
  protected:
    Entry *m_entry;                                         ///< Wrapped Entry pointer.
  }; /* class Handle */

} /* TransformationTypes */

#pragma once

#include <string>
#include <vector>

#include "taintflag.hh"
#include "Data.hh"
#include "TransformationEntry.hh"
#include "InputHandle.hh"
#include "OutputHandle.hh"

template <typename FloatType> class SingleOutputT;

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
  template<typename SourceFloatType, typename SinkFloatType>
  class HandleT {
  public:
    using HandleType       = HandleT<SourceFloatType,SinkFloatType>;
    using EntryType        = EntryT<SourceFloatType,SinkFloatType>;
    using OutputHandleType = OutputHandleT<SinkFloatType>;
    using InputHandleType  = InputHandleT<SourceFloatType>;
    using DataType         = Data<SinkFloatType>;
    using SingleOutputType = SingleOutputT<SourceFloatType>;

    HandleT(): m_entry(nullptr) { }                                 ///< Default constructor.
    HandleT(EntryType &entry) : m_entry(&entry) { }                 ///< Constructor. @param entry -- an Entry instance to wrap.
    HandleT(const HandleType &other): HandleT(*other.m_entry) { }   ///< Constructor. @param other -- Handle instance to get Entry to wrap.

    const std::string &name() const { return m_entry->name; }                            ///< Get entry name.
    const std::string &label() const { return m_entry->attrs["_label"]; }                ///< Get entry label.
    void  setLabel(const std::string& label) const { m_entry->attrs["_label"]=label; }   ///< Set entry label.
    std::vector<InputHandleType> inputs() const;                                         ///< Get vector of inputs.
    std::vector<OutputHandleType> outputs() const;                                       ///< Get vector of outputs.

    InputHandleType input(const std::string &name);                                      ///< Add named input.
    InputHandleType input(const std::string &name, int mapto);                           ///< Add named input.

    InputHandleType input(SingleOutputType &output);                                     ///< Create a new input and connect to the SingleOutput transformation.
    InputHandleType input(const std::string &name, SingleOutputType &output);            ///< Create a new input and connect to the SingleOutput transformation.

    InputHandleType input(SingleOutputType &output, int mapto);                          ///< Create a new input and connect to the SingleOutput transformation.
    InputHandleType input(const std::string &name, SingleOutputType &output, int mapto); ///< Create a new input and connect to the SingleOutput transformation.

    /**
     * @brief Add new named output.
     *
     * @param name -- new Sink's name.
     * @return OutputHandle for the new Sink.
     */
    OutputHandleType output(const std::string &name) {
      return m_entry->addSink(name);
    }
    OutputHandleType output(SingleOutputType &output);               ///< Create a new output with a same name as SingleOutput's output.

    /**
     * @brief Return i-th Entry's Sink's data.
     *
     * The transformation function is evaluated if needed.
     *
     * @param i -- index of an output.
     * @return Data instance.
     */
    const DataType &operator[](int i) const { return m_entry->data(i); }

    /**
     * @brief Trigger an update of an Entry by simulating access to the i-th data.
     *
     * @param i -- Entry's Sink's index.
     */
    void update(int i) const { (void)m_entry->data(i); }
    void touch() const { m_entry->touch_global(); }         ///< Call Entry::touch_global(). @copydoc Entry::touch()
    void touch_local() const { m_entry->touch(); }          ///< Call Entry::touch(). @copydoc Entry::touch() // FIXME: should be removed, but needed by TreeManager
    void touchTypes() { m_entry->touchTypes(); }            ///< Call Entry::touchTypes(). @copydoc Entry::touchTypes()
    void updateTypes() { m_entry->updateTypes(); }          ///< Call Entry::updateTypes(). @copydoc Entry::updateTypes()

#ifdef GNA_CUDA_SUPPORT
    void setLocation(DataLocation loc) { m_entry->setLocation(loc); m_entry->updateTypes(); } ///< Change Entry location
#endif

    void unfreeze() { m_entry->tainted.unfreeze(); }        ///< Unfreeze Entry's taintflag.

    void taint() { m_entry->tainted.taint(); }              ///< Taint the Entry's taintflag. The outputs will be evaluated upon request.
    bool tainted() { return m_entry->tainted; }             ///< Return the Entry's taintflag status.
    taintflag& getTaintflag() const noexcept { return m_entry->tainted; } ///< Return taintflag of underlying Entry

    void readVariables(ParametrizedTypes::ParametrizedBase* parbase);  ///< Set parbase and read the variables
    void readVariables();                                              ///< Read the variables from previously set ParametrizedBase

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
    EntryType *m_entry;                                     ///< Wrapped Entry pointer.
  }; /* class Handle */
} /* TransformationTypes */

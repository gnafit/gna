#pragma once

#include <string>
#include <memory>
#include <type_traits>
#include <boost/noncopyable.hpp>

#include "Data.hh"
#include "variable.hh"
#include "taintflag.hh"
#include "TransformationFunction.hh"
#include "FunctionDescriptor.hh"

#include "Source.hh"
#include "Sink.hh"
#include "Storage.hh"

namespace TransformationTypes
{
  template<typename SourceFloatType, typename SinkFloatType> class BaseT;
  template<typename FloatType> class InputHandleT;
  template<typename FloatType> class OutputHandleT;

  template<typename FloatType>
  using SourcesContainerT  = boost::ptr_vector<SourceT<FloatType>>;   ///< Container for Source pointers.

  template<typename FloatType>
  using SinksContainerT    = boost::ptr_vector<SinkT<FloatType>>;     ///< Container for Sink pointers.

  template<typename FloatType>
  using StoragesContainerT = boost::ptr_vector<StorageT<FloatType>>;  ///< Container for Storage pointers.

  /**
   * @brief Definition of a single transformation.
   *
   * Entry defines a transformation that:
   *   - has zero or more inputs: Source instances.
   *   - has one or more outputs: Sink instances.
   *   - has a function Entry::fun that defines the transformation.
   *   - may have several type functions (Entry::typefuns), that check the
   *     input types, derive the output types and the types of internal data to be allocated.
   *
   * Entry has a taintflag (Entry::taintflag), then defines whether the Entry's Sink instances
   * contain up to date output data.
   *
   * Entry will call the transformation function Entry::fun before returning
   * Data in case Entry is tainted or any of the Inputs is tainted.
   *
   * Entry::fun may have several different implementations that can be used for example to define
   * the GPU version of the transformation. The implementations are named and
   * stored in the Entry::functions container. The active function Entry::fun may be switched to any
   * function in Entry::functions via Entry::switchFunction() method.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename SourceFloatType, typename SinkFloatType=SourceFloatType>
  struct EntryT: public boost::noncopyable {
    using BaseType                           = BaseT<SourceFloatType,SinkFloatType>;
    using EntryType                          = EntryT<SourceFloatType,SinkFloatType>;
    using StorageFloatType                   = SourceFloatType;

    using SourceDataType                     = Data<SourceFloatType>;
    using SinkDataType                       = Data<SinkFloatType>;
    using StorageDataType                    = Data<StorageFloatType>;

    using SourceType                         = SourceT<SourceFloatType>;
    using SourcesContainerType               = SourcesContainerT<SourceFloatType>;
    using InputHandleType                    = InputHandleT<SourceFloatType>;

    using SinkType                           = SinkT<SinkFloatType>;
    using SinksContainerType                 = SinksContainerT<SinkFloatType>;
    using OutputHandleType                   = OutputHandleT<SinkFloatType>;

    using StorageType                        = StorageT<StorageFloatType>;
    using StoragesContainerType              = StoragesContainerT<StorageFloatType>;

    using FunctionArgsType                   = FunctionArgsT<StorageFloatType,SinkFloatType>;
    using FunctionArgsPtr                    = std::unique_ptr<FunctionArgsType>;

    using TypesFunctionArgsType              = TypesFunctionArgsT<StorageFloatType,SinkFloatType>;
    using StorageTypesFunctionArgsType       = StorageTypesFunctionArgsT<StorageFloatType,SinkFloatType>;

    using FunctionType                       = FunctionT<StorageFloatType,SinkFloatType>;
    using TypesFunctionType                  = TypesFunctionT<StorageFloatType,SinkFloatType>;
    using StorageTypesFunctionType           = StorageTypesFunctionT<StorageFloatType,SinkFloatType>;

    using TypesFunctionsContainerType        = TypesFunctionsContainerT<StorageFloatType,SinkFloatType>;
    using StorageTypesFunctionsContainerType = StorageTypesFunctionsContainerT<StorageFloatType,SinkFloatType>;

    using FunctionDescriptorType             = FunctionDescriptorT<SourceFloatType,SinkFloatType>;
    using FunctionDescriptorsContainerType   = FunctionDescriptorsContainerT<SourceFloatType,SinkFloatType>;

    EntryT(const std::string &name, const BaseType *parent); ///< Constructor.
    EntryT(const EntryType &other, const BaseType *parent);  ///< Clone constructor.
    ~EntryT();                                           ///< Destructor.

    InputHandleType addSource(const std::string &name, bool inactive=false);      ///< Initialize and return new Source.
    OutputHandleType addSink(const std::string &name);                            ///< Initialize and return new Sink.

    void evaluate();                                     ///< Do actual calculation by calling Entry::fun.
    void update();                                       ///< Do actual calculation by calling Entry::fun via evaluate() and resets the taintflag.
    void evaluateTypes();                                ///< Evaluate output types based on input types via Entry::typefuns call, allocate memory.
    void updateTypes();                                  ///< Evaluate output types based on input types via Entry::typefuns call, allocate memory.

    void touch();                                        ///< Update the transformation if it is not frozen and tainted.
    const SinkDataType &data(int i);                     ///< Evaluates the function if needed and returns i-th data.

    bool check() const;                                  ///< Checks that Data are initialized.
    void dump(size_t level = 0) const;                   ///< Recursively print Source names and their connection status.

    // Names, labels and meta
    std::string name;                                    ///< Transformation name.
    std::string label;                                   ///< Transformation label.
    const BaseType *parent;                              ///< Base class, containing the transformation Entry.

    // Data
    SourcesContainerType sources;                        ///< Transformation inputs (sources).
    SinksContainerType sinks;                            ///< Transformation outputs (sinks).
    StoragesContainerType storages;                      ///< Transformation internal Storage instances.
    std::vector<size_t> mapping;                         ///< Inputs to outputs mapping (for multi-transformations).

    // Functions
    FunctionType fun=nullptr;                            ///< The function that does actual calculation.
    TypesFunctionsContainerType typefuns;                ///< Vector of TypeFunction objects.
    FunctionDescriptorsContainerType functions;          ///< Map with FunctionDescriptor instances, containing several Function implementations.
    std::string funcname;                                ///< Active Function name.

    // Status
    taintflag tainted;                                   ///< taintflag shows whether the result is up to date.
    int initializing;                                    ///< Initialization status. initializing>0 when Entry is being configured via Initializer.

    // Function args
    FunctionArgsPtr functionargs;                        ///< Transformation function arguments.

    void switchFunction(const std::string& name);        ///< Use Function `name` as Entry::fun.
  private:
    template <typename InsT, typename OutsT>
    void initSourcesSinks(const InsT &inputs, const OutsT &outputs); ///< Initialize the Data for inputs and outputs.

    void initInternals(StorageTypesFunctionArgsType& fargs);         ///< Initialize the Data for the internal storage.
  }; /* struct Entry */
} /* TransformationTypes */

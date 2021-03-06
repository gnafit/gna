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

#include "config_vars.h"

namespace GNA{
  template<typename FloatType> class TreeManager;
}

namespace TypeClasses{
  template<typename FloatType> class TypeClassT;
}

namespace ParametrizedTypes{
  class ParametrizedBase;
}

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

  template<typename FloatType>
  using TypeClassContainerT = boost::ptr_vector<TypeClasses::TypeClassT<FloatType>>;  ///< Container for TypeClass pointers.

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
  class EntryT: public boost::noncopyable {
  public:
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

    using TypeClassContainerType             = TypeClassContainerT<SourceFloatType>;

    using TreeManagerType                    = GNA::TreeManager<SourceFloatType>;

    EntryT(const std::string &name, const BaseType *parent); ///< Constructor.
    //EntryT(const EntryType &other, const BaseType *parent);  ///< Clone constructor.
    ~EntryT();                                               ///< Destructor.

    InputHandleType addSource(const std::string &name, bool inactive=false);      ///< Initialize and return new Source.
    OutputHandleType addSink(const std::string &name);                            ///< Initialize and return new Sink.

    void evaluate();                                     ///< Do actual calculation by calling Entry::fun.
    void update();                                       ///< Do actual calculation by calling Entry::fun via evaluate() and resets the taintflag.
    void touchTypes();                                   ///< If Needed, evaluate output types based on input types via Entry::typefuns call, allocate memory.
    void updateTypes();                                  ///< Evaluate output types based on input types via Entry::typefuns call, allocate memory.
    void finalize();                                     ///< Called on initialization to indicate, that no inputs are expected, but TypeFunctions should be evaluated.

    void touch();                                        ///< Update the transformation if it is not frozen and tainted.
    void touch_global();                                 ///< Update the transformation if it is not frozen and tainted.
    const SinkDataType &data(int i);                     ///< Evaluates the function if needed and returns i-th data.

    bool check() const;                                  ///< Checks that Data are initialized.
    void dump(size_t level = 0) const;                   ///< Recursively print Source names and their connection status.

    // Names, labels and meta
    std::string name;                                    ///< Transformation name.
    Attrs attrs;                                         ///< Map with entry attributes
    const BaseType *parent;                              ///< Base class, containing the transformation Entry.

    // Data
    SourcesContainerType sources;                        ///< Transformation inputs (sources).
    SinksContainerType sinks;                            ///< Transformation outputs (sinks).
    StoragesContainerType storages;                      ///< Transformation internal Storage instances.
    std::vector<size_t> mapping;                         ///< Inputs to outputs mapping (for multi-transformations).

    // Functions
    FunctionType fun=nullptr;                            ///< The function that does actual calculation.
    TypesFunctionsContainerType typefuns;                ///< Vector of TypeFunction objects.
    TypeClassContainerType typeclasses;                  ///< Vector of TypeClass instances
    FunctionDescriptorsContainerType functions;          ///< Map with FunctionDescriptor instances, containing several Function implementations.
    std::string funcname;                                ///< Active Function name.

#ifdef GNA_CUDA_SUPPORT
    void setEntryLocation(DataLocation loc);            ///< Sets the target (Host or Device) for execution of current transformation
    void setEntryDataLocation(DataLocation loc);
    DataLocation getEntryLocation() const;              ///<  Returns the target (Host or Device) for execution of current transformation
#endif

    // Status
    taintflag tainted;                                   ///< taintflag shows whether the result is up to date.
    taintflag typestainted;                              ///< taintflag shows whether the datatype is up to date.
    bool finalized=false;                                ///< TypeFunctions should be evaluated even if there are no inputs.

    // Function args
    FunctionArgsPtr functionargs;                        ///< Transformation function arguments.

    bool switchFunction(const std::string& name, bool strict=true); ///< Use Function `name` as Entry::fun.
    bool initFunction(const std::string& name, bool strict=true);   ///< Use Function `name` as Entry::fun. Does not update types.

    size_t hash() const { return reinterpret_cast<size_t>((void*)this); } ///< Return entry address as size_t

    TreeManagerType* m_tmanager=nullptr;                 ///< Tree manager

#ifdef GNA_CUDA_SUPPORT
    void setLocation(DataLocation loc) { m_entryLoc=loc; } ///< Change Entry location
#endif
  private:
#ifdef GNA_CUDA_SUPPORT
    DataLocation m_entryLoc = DataLocation::Host;       ///< In case of GPU support is swiched on, the target for execution(Host or Device). Host by default.
#endif

    template <typename InsT, typename OutsT>
    void initSourcesSinks(const InsT &inputs, const OutsT &outputs); ///< Initialize the Data for inputs and outputs.

    void initInternals(StorageTypesFunctionArgsType& fargs);         ///< Initialize the Data for the internal storage.

  }; /* class Entry */
} /* TransformationTypes */

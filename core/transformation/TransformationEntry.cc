#include "TransformationEntry.hh"

#include <set>
#include <string>
#include <stdexcept>
#include <algorithm>

#include "TransformationFunctionArgs.hh"
#include "Source.hh"
#include "InputHandle.hh"
#include "OutputHandle.hh"
#include "Atypes.hh"
#include "TransformationErrors.hh"
#include "TypeClasses.hh"
#include "GPUFunctionArgs.hh"
#include "TreeManager.hh"

#include "config_vars.h"

using TransformationTypes::BaseT;
using TransformationTypes::EntryT;
using TransformationTypes::AtypesT;
using TransformationTypes::OutputHandleT;
using TransformationTypes::InputHandleT;
using TransformationTypes::TypeError;

template<typename FloatType>
using TypeClassT = TypeClasses::TypeClassT<FloatType>;

/**
 * @brief Constructor.
 *
 * @param name -- Entry's name.
 * @param parent -- Base class instance to hold the Entry.
 */
template<typename SourceFloatType, typename SinkFloatType>
EntryT<SourceFloatType,SinkFloatType>::EntryT(const std::string &name, const BaseT<SourceFloatType,SinkFloatType> *parent)
  : name(name), attrs({{"_label", name}}), parent(parent), tainted(name.c_str()),
    functionargs(new FunctionArgsType(this)), m_tmanager(GNA::TreeManager<SourceFloatType>::current())
{
  if(m_tmanager){
    m_tmanager->registerTransformation(this);
  }
}

/**
 * @brief Clone constructor.
 * @todo Cross check the description.
 *
 * @param other -- Entry to copy name, inputs and outputs from.
 * @param parent -- Base class instance to hold the Entry.
 */
//template<typename SourceFloatType, typename SinkFloatType>
//EntryT<SourceFloatType,SinkFloatType>::EntryT(const EntryT<SourceFloatType,SinkFloatType> &other, const BaseT<SourceFloatType,SinkFloatType> *parent)
  //: name(other.name), attrs(other.attrs), parent(parent),
    //sources(other.sources.size()), sinks(other.sinks.size()),
    //mapping(other.mapping),
    //fun(), typefuns(), typeclasses(), tainted(other.name.c_str()),
    //functionargs(new FunctionArgsType(this)),
    //m_tmanager(other.m_tmanager)
//{
  //initSourcesSinks(other.sources, other.sinks);
//}


/**
 * @brief Destructor.
 *
 * The destructor is defined explicitly in order to enable unique_ptr members to delete their objects.
 */
template<typename SourceFloatType, typename SinkFloatType>
EntryT<SourceFloatType,SinkFloatType>::~EntryT<SourceFloatType,SinkFloatType>(){}

/**
 * @brief Initialize the clones for inputs and outputs.
 *
 * To be used in a copy constructor Entry::Entry().
 *
 * @tparam InsT -- container type for Source instances.
 * @tparam OutsT -- container type for Sink instances.
 * @param inputs -- container for Source instances.
 * @param outputs -- container for Sink instances.
 */
template<typename SourceFloatType, typename SinkFloatType>
template <typename InsT, typename OutsT>
void EntryT<SourceFloatType,SinkFloatType>::initSourcesSinks(const InsT &inputs, const OutsT &outputs) {
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(sources),
                 [this](const SourceType &s) { return new SourceType{s, this}; });
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(sinks),
                 [this](const SinkType &s) { return new SinkType{s, this}; });
  functionargs->updateTypes();
}

/**
 * @brief Initialize and return new input Source.
 *
 * @param name -- new Source's name.
 * @param inactive -- set source inactive (do not subscribe taintflag)
 * @return InputHandle for the new Source.
 */
template<typename SourceFloatType, typename SinkFloatType>
InputHandleT<SourceFloatType> EntryT<SourceFloatType,SinkFloatType>::addSource(const std::string &name, bool inactive) {
  auto *s = new SourceType(name, this, inactive);
  sources.push_back(s);
  return InputHandleType(*s);
}

/**
 * @brief Initialize and return new input Sink.
 *
 * @param name -- new Sink's name.
 * @return Source for the new Sink.
 */
template<typename SourceFloatType, typename SinkFloatType>
OutputHandleT<SinkFloatType> EntryT<SourceFloatType,SinkFloatType>::addSink(const std::string &name) {
  auto *s = new SinkType(name, this);
  sinks.push_back(s);
  return OutputHandleType(*s);
}

/**
 * @brief Checks that Data are initialized.
 *
 * Checks that all the Source instances are materialized (Data are allocated).
 * The procedure triggers Entry::check() for all the preceding Entry instances.
 *
 * @return true if everything is OK.
 */
template<typename SourceFloatType, typename SinkFloatType>
bool EntryT<SourceFloatType,SinkFloatType>::check() const {
  for (const SourceType &s: sources) {
    if (!s.materialized()) {
      return false;
    }
  }
  for (const SourceType &s: sources) {
    if (!s.sink->entry->check()) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Do actual calculation by calling Entry::fun.
 *
 * Does not reset the taintflag.
 */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::evaluate() {
  return fun(*functionargs);
}

/**
 * @brief Do actual calculation by calling Entry::fun via evaluate() and resets the taintflag.
 *
 * Handles exception and resets the taintflag.
 */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::update() {
  Status status = Status::Success;
  try {
    evaluate();
#ifdef GNA_CUDA_SUPPORT
    setEntryDataLocation(m_entryLoc);
#endif
    tainted = false;
  } catch (const SinkTypeError<SinkType>&) {
    status = Status::Failed;
  }
  for (auto &sink: sinks) {
    sink.data->state = status;
  }
}

/** @brief Recursively print Source names and their connection status.
 *
 * For each Source's Entry Entry::dump() method is called with hither indentation level.
 *
 * @param level -- indentation level.
 */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::dump(size_t level) const {
  std::string spacing = std::string(level*2, ' ');
  std::cerr << spacing << "Transformation " << name << std::endl;
  if (sources.empty()) {
    std::cerr << spacing << "no inputs" << std::endl;
  }
  for (const SourceType &s: sources) {
    std::cerr << spacing << "Input " << s.name;
    if (!s.sink) {
      std::cerr << " NOT CONNECTED" << std::endl;
    } else {
      std::cerr << " connected to " << s.sink->name << std::endl;
      s.sink->entry->dump(level+1);
    }
  }
}

/**
 * @brief Evaluate output types based on input types, allocate memory.
 *
 * evaluateTypes() calls:
 *   - each of the TypeFunction functions;
 *   - each of the current functions StorageTypesFunction functions.
 *
 * TypeFunction may do the following:
 *   - determine the consistency of the input data types;
 *   - derive the output data types;
 *   - derive the necessary Storage data types, that are common for all the Entry::functions.
 *
 * A list of StorageTypesFunction functions is associated with each function
 * from the Entry::functions. Different transformation implementation may require different
 * internal storage to be allocated. StorateTypesFunctions may derive the necessary internal
 * storage to be allocated.
 *
 * In case any output DataType has been changed or created:
 *   - the corresponding Data instance is created. Memory allocation happens here.
 *   - if sources are connected further, the subsequent Entry::evaluateTypes() are
 *   also executed.
 *
 * If CUDA enabled, allocates memory for sources (in case it wasn't allocated earlier) and sinks.
 *
 * @todo DataType instances created within StorageTypesFunction will trigger data reallocation
 * in any case. Should be fixed.
 *
 * @exception std::runtime_error in case any of type functions fails.
 */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::evaluateTypes() {
  TypesFunctionArgsType fargs(this);
  StorageTypesFunctionArgsType sargs(fargs);
  bool success = false;
  TR_DPRINTF("evaluating types for %s: \n", name.c_str());
  try {
    for (auto &typefun: typefuns) {
      typefun(fargs);
    }
    for (auto &typeclass: typeclasses) {
      typeclass.processTypes(fargs);
    }
    auto& itypefuns=functions[funcname].typefuns;
    for (auto &typefun: itypefuns) {
      typefun(sargs);
    }
    success = true;
  } catch (const TypeError &exc) {
    TR_DPRINTF("types[%s]: failed\n", name.c_str());
    throw std::runtime_error(
      (fmt::format("Transformation: type updates failed for `{0}': {1}", name, exc.what())));
  } catch (const typename AtypesT<SourceFloatType,SinkFloatType>::Undefined&) {
    TR_DPRINTF("types[%s]: undefined\n", name.c_str());
  }
  if (success) {
    std::set<EntryType*> deps;
    TR_DPRINTF("types[%s]: success\n", name.c_str());
    auto& rets=fargs.rets;
    for (size_t i = 0; i < sinks.size(); ++i) {
      auto& ret  = rets[i];
      auto& sink = sinks[i];
      if (!ret.buffer && sink.data && !ret.requiresReallocation(sink.data->type)) {
        continue;
      }
      if (ret.defined()) {
        sink.data.reset(new SinkDataType(ret));
      }
      else{
        sink.data.reset();
      }
      TR_DPRINTF("types[%s, %s]: ", name.c_str(), sink.name.c_str());
#ifdef TRANSFORMATION_DEBUG
      sink.data->type.dump();
#endif // TRANSFORMATION_DEBUG
      for (auto *depsrc: sink.sources) {
        deps.insert(depsrc->entry);
      }
    }
    for (auto *dep: deps) {
      dep->evaluateTypes();
    }
    initInternals(sargs);

#ifdef GNA_CUDA_SUPPORT
    // GPU: require GPU memory for previous transformation's sink
    if (this->getEntryLocation() == DataLocation::Device){
      for (auto &source : sources) {
          source.requireGPU();
      }
      for (auto &sink : sinks) {
        sink.requireGPU(this->getEntryLocation());
      }
      for (auto &intern : storages) {
        intern.requireGPU(this->getEntryLocation());
      }
      // init gpu storage
    }
#endif
  }

  /// TODO: do it optionally
  functionargs->requireGPU();
  functionargs->updateTypes();
}

/** @brief Evaluate output types based on input types via Entry::typefuns call, allocate memory. */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::updateTypes() {
  evaluateTypes();
}

/** @brief Update the transformation if it is not frozen and tainted. */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::touch() {
  if (tainted) {
    update();
  }
}

/** @brief Update the transformation if it is not frozen and tainted. */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::touch_global() {
  if (tainted) {
    if(m_tmanager){
      m_tmanager->update();
    }
    update();
  }
}

/**
 * Returns i-th data. Does the calculation if needed.
 * If CUDA enabled and relevant data is placed on GPU, it synchronizes data before return it.
 * @param i -- index of a Sink to read the data.
 * @return i-th Sink's Data.
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case input data is undefined.
 */
template<typename SourceFloatType, typename SinkFloatType>
const Data<SinkFloatType> &EntryT<SourceFloatType,SinkFloatType>::data(int i) {
  if (i < 0 or static_cast<size_t>(i) > sinks.size()) {
    auto msg = fmt::format("invalid sink idx {0}, have {1} sinks", i, sinks.size());
    throw CalculationError<EntryType>(this, msg);
  }
  const SinkType &sink = sinks[i];
  if (!sink.data) {
    auto msg = fmt::format("sink {0} ({1}) have no type", i, sink.name);
    throw CalculationError<EntryType>(this, msg);
  }
  touch();
#ifdef GNA_CUDA_SUPPORT
  if (m_entryLoc==DataLocation::Device && sink.data->gpuArr) {
    sink.data->gpuArr->sync( DataLocation::Host );
  }
#endif
  return *sink.data;
}

/**
 * @brief Use Function `name` as Entry::fun.
 *
 * The method replaces the transformation function Entry::fun with another function from the
 * Entry::functions map. The function triggers Entry::evaluateTypes() function.
 *
 * @param name -- function name.
 * @exception std::runtime_error in case the function `name` does not exist.
 */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::switchFunction(const std::string& name){
  initFunction(name);
  evaluateTypes();
}

/**
 * @brief Use Function `name` as Entry::fun.
 *
 * The method replaces the transformation function Entry::fun with another function from the
 * Entry::functions map.
 *
 * @param name -- function name.
 * @exception std::runtime_error in case the function `name` does not exist.
 */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::initFunction(const std::string& name){
  auto it = functions.find(name);
  if(it==functions.end()){
    auto msg = fmt::format("invalid function name {0}", name.data());
    throw std::runtime_error(msg);
  }
  fun = it->second.fun;
#ifdef GNA_CUDA_SUPPORT
  setEntryLocation(it->second.funcLoc);
#endif
  funcname=name;
//  evaluateTypes();
}


#ifdef GNA_CUDA_SUPPORT
/**
 *    @brief Sets the target (Host or Device) for execution of current transformation
 */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::setEntryLocation(DataLocation loc) {
    m_entryLoc = loc;
}

/**
 *    @brief Sets the target (Host or Device) for transformation sinks
 *    \warning Be careful! It changes sink location values directly without synchronization invoke!
 */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::setEntryDataLocation(DataLocation loc) {
    for (const SinkType &s: sinks) {
        s.data->gpuArr->setLocation(loc);
    }
}

/**
 * @brief Returns the target (Host or Device) for execution of current transformation
 */
template<typename SourceFloatType, typename SinkFloatType>
DataLocation EntryT<SourceFloatType,SinkFloatType>::getEntryLocation() const {
            return m_entryLoc;
}
#endif



/**
 * @brief Initialize the Data for the internal storage.
 *
 * Initializes the Data instance with proper shape for each DataType in Itypes.
 *
 * @param fargs -- Storage TypesFunction arguments.
 */
template<typename SourceFloatType, typename SinkFloatType>
void EntryT<SourceFloatType,SinkFloatType>::initInternals(StorageTypesFunctionArgsT<SourceFloatType,SinkFloatType>& fargs){
  auto& itypes=fargs.ints;

  // Truncate storages in case less storages is required
  if(storages.size()>itypes.size()){
    storages.resize(itypes.size());
  }
  for (size_t i(0); i<itypes.size(); ++i) {
    auto& int_dtype=itypes[i];

    // in case the Storage is allocated for current index
    // check if new allocation is unnecessary
    if(i<storages.size()){
      auto& storage = storages[i];
      if (!int_dtype.buffer && storage.data && storage.data->type==int_dtype) {
        continue;
      }
    }

    // create new storage and allocate memory (if needed)
    auto* newstorage = new StorageType(this);
    newstorage->data.reset(new Data<SourceFloatType>(int_dtype));

    // debug
    TR_DPRINTF("stypes[%s, %i %s]: ", name.c_str(), static_cast<int>(i), newstorage->name.c_str());
#ifdef TRANSFORMATION_DEBUG
    int_dtype.dump();
#endif // TRANSFORMATION_DEBUG

    // replace existing Storage or extend the storages
    if(i<storages.size()){
      storages.replace(i, newstorage);
    }
    else{
      storages.push_back(newstorage);
    }
  }
}

template class TransformationTypes::EntryT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::EntryT<float>;
#endif

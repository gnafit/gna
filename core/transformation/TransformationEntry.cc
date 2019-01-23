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
#include "Rtypes.hh"
#include "TransformationErrors.hh"
#include "GPUFunctionArgs.hh"

using TransformationTypes::Base;
using TransformationTypes::Entry;

using TransformationTypes::Atypes;
using TransformationTypes::Rtypes;

using TransformationTypes::Source;
using TransformationTypes::Sink;
using TransformationTypes::OutputHandle;
using TransformationTypes::InputHandle;

using TransformationTypes::TypeError;

/**
 * @brief Constructor.
 *
 * @param name -- Entry's name.
 * @param parent -- Base class instance to hold the Entry.
 */
Entry::Entry(const std::string &name, const Base *parent)
  : name(name), label(name), parent(parent), tainted(name.c_str()), initializing(0),
    cpuargs(new FunctionArgs(this)),
    gpuargs(new GPUFunctionArgs(this))
{ }

/**
 * @brief Clone constructor.
 * @todo Cross check the description.
 *
 * @param other -- Entry to copy name, inputs and outputs from.
 * @param parent -- Base class instance to hold the Entry.
 */
Entry::Entry(const Entry &other, const Base *parent)
  : name(other.name), label(other.label), parent(parent),
    sources(other.sources.size()), sinks(other.sinks.size()),
    fun(), typefuns(), tainted(other.name.c_str()), initializing(0),
    cpuargs(new FunctionArgs(this)),
    gpuargs(new GPUFunctionArgs(this))
{
  initSourcesSinks(other.sources, other.sinks);
}


/**
 * @brief Destructor.
 *
 * The destructor is defined explicitly in order to enable unique_ptr members to delete their objects.
 */
Entry::~Entry(){}

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
template <typename InsT, typename OutsT>
void Entry::initSourcesSinks(const InsT &inputs, const OutsT &outputs) {
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(sources),
                 [this](const Source &s) { return new Source{s, this}; });
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(sinks),
                 [this](const Sink &s) { return new Sink{s, this}; });
  gpuargs->evaluateTypes();
}

/**
 * @brief Initialize and return new input Source.
 *
 * @param name -- new Source's name.
 * @param inactive -- set source inactive (do not subscribe taintflag)
 * @return InputHandle for the new Source.
 */
InputHandle Entry::addSource(const std::string &name, bool inactive) {
  auto *s = new Source(name, this, inactive);
  sources.push_back(s);
  return InputHandle(*s);
}

/**
 * @brief Initialize and return new input Sink.
 *
 * @param name -- new Sink's name.
 * @return Source for the new Sink.
 */
OutputHandle Entry::addSink(const std::string &name) {
  auto *s = new Sink(name, this);
  sinks.push_back(s);
  return OutputHandle(*s);
}

/**
 * @brief Checks that Data are initialized.
 *
 * Checks that all the Source instances are materialized (Data are allocated).
 * The procedure triggers Entry::check() for all the preceding Entry instances.
 *
 * @return true if everything is OK.
 */
bool Entry::check() const {
  for (const Source &s: sources) {
    if (!s.materialized()) {
      return false;
    }
  }
  for (const Source &s: sources) {
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
void Entry::evaluate() {
  return fun(*cpuargs);
}

/**
 * @brief Do actual calculation by calling Entry::fun via evaluate() and resets the taintflag.
 *
 * Handles exception and resets the taintflag.
 */
void Entry::update() {
  Status status = Status::Success;
  try {
    evaluate();
    tainted = false;
  } catch (const SinkTypeError&) {
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
void Entry::dump(size_t level) const {
  std::string spacing = std::string(level*2, ' ');
  std::cerr << spacing << "Transformation " << name << std::endl;
  if (sources.empty()) {
    std::cerr << spacing << "no inputs" << std::endl;
  }
  for (const Source &s: sources) {
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
void Entry::evaluateTypes() {
  TypesFunctionArgs fargs(this);
  StorageTypesFunctionArgs sargs(fargs);
  bool success = false;
  TR_DPRINTF("evaluating types for %s: \n", name.c_str());
  try {
    for (auto &typefun: typefuns) {
      typefun(fargs);
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
  } catch (const Atypes::Undefined&) {
    TR_DPRINTF("types[%s]: undefined\n", name.c_str());
  }
  if (success) {
    std::set<Entry*> deps;
    TR_DPRINTF("types[%s]: success\n", name.c_str());
    auto& rets=fargs.rets;
    for (size_t i = 0; i < sinks.size(); ++i) {
      auto& ret  = rets[i];
      auto& sink = sinks[i];
      if (!ret.buffer && sink.data && sink.data->type==ret) {
        continue;
      }
      if (ret.defined()) {
        sink.data.reset(new Data<double>(ret));
      }
      else{
        sink.data.reset();
      }
      TR_DPRINTF("types[%s, %s]: ", name.c_str(), sink.name.c_str());
#ifdef TRANSFORMATION_DEBUG
      sink.data->type.dump();
#endif // TRANSFORMATION_DEBUG
      for (Source *depsrc: sink.sources) {
        deps.insert(depsrc->entry);
      }
    }

    for (Entry *dep: deps) {
      dep->evaluateTypes();
    }
    initInternals(sargs);


    // GPU: require GPU memory for previous transformation's sink
#ifdef GNA_CUDA_SUPPORT
    if (this->getEntryLocation() == DataLocation::Device) {
      for (auto &source : sources) {
          source.sink->data->require_gpu();
	 //  source.sink->data->gpuArr->setLocation( this->getEntryLocation() );
      }
      for (auto &sink : sinks) {
        sink.data->require_gpu();
        sink.data->gpuArr->setLocation( this->getEntryLocation() );
      }
      gpustorage = new GPUStorage(this); 
      std::cerr << "afret gpu storage new" << std::endl;
      gpustorage->initGPUStorage();
    }
#endif

  }

  gpuargs->evaluateTypes();
}

/** @brief Evaluate output types based on input types via Entry::typefuns call, allocate memory. */
void Entry::updateTypes() {
  evaluateTypes();
}

/** @brief Update the transformation if it is not frozen and tainted. */
void Entry::touch() {
  if (tainted) {
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
const Data<double> &Entry::data(int i) {
  if (i < 0 or static_cast<size_t>(i) > sinks.size()) {
    auto msg = fmt::format("invalid sink idx {0}, have {1} sinks", i, sinks.size());
    throw CalculationError(this, msg);
  }
  const Sink &sink = sinks[i];
  if (!sink.data) {
    auto msg = fmt::format("sink {0} ({1}) have no type", i, sink.name);
    throw CalculationError(this, msg);
  }
  touch();
#ifdef GNA_CUDA_SUPPORT
  if (sink.data->gpuArr != nullptr) {
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
void Entry::switchFunction(const std::string& name){
  auto it = functions.find(name);
  if(it==functions.end()){
    auto msg = fmt::format("invalid function name {0}", name.data());
    throw std::runtime_error(msg);
  }
  fun = it->second.fun;

  funcname=name;
  evaluateTypes();
}

/**
 * @brief Initialize the Data for the internal storage.
 *
 * Initializes the Data instance with proper shape for each DataType in Itypes.
 *
 * @param fargs -- Storage TypesFunction arguments.
 */
void Entry::initInternals(StorageTypesFunctionArgs& fargs){
  storages.clear();
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
    auto* newstorage = new Storage(this);
    newstorage->data.reset(new Data<double>(int_dtype));

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

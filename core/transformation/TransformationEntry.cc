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
  : name(name), label(name), parent(parent), initializing(0), frozen(false)
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
    fun(), typefuns(), initializing(0), frozen(false)
{
  initSourcesSinks(other.sources, other.sinks);
}

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
}

/**
 * @brief Initialize and return new input Source.
 *
 * @param name -- new Source's name.
 * @return InputHandle for the new Source.
 */
InputHandle Entry::addSource(const std::string &name) {
  Source *s = new Source(name, this);
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
  Sink *s = new Sink(name, this);
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
  auto fargs = FunctionArgs(this);
  return fun(fargs);
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
 * @brief Evaluate output types based on input types via Entry::typefuns call, allocate memory.
 *
 * evaluateTypes() calls each of the TypeFunction functions which determine
 * the consistency of the inputs and derive the types (DataType) of the outputs.
 *
 * In case any output DataType has been changed or created:
 *   - the corresponding Data instance is created. Memory allocation happens here.
 *   - if sources are connected further, the subsequent Entry::evaluateTypes() are
 *   also executed.
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
      (format("Transformation: type updates failed for `%1%': %2%") % name % exc.what()).str()
      );
  } catch (const Atypes::Undefined&) {
    TR_DPRINTF("types[%s]: undefined\n", name.c_str());
  }
  if (success) {
    std::set<Entry*> deps;
    TR_DPRINTF("types[%s]: success\n", name.c_str());
    auto& rets=fargs.rets;
    for (size_t i = 0; i < sinks.size(); ++i) {
      if (!rets[i].buffer && sinks[i].data && sinks[i].data->type == rets[i]) {
        continue;
      }
      if (rets[i].defined()) {
        sinks[i].data.reset(new Data<double>(rets[i]));
      }
      else{
        sinks[i].data.reset();
      }
      TR_DPRINTF("types[%s, %s]: ", name.c_str(), sinks[i].name.c_str());
#ifdef TRANSFORMATION_DEBUG
      sinks[i].data->type.dump();
#endif // TRANSFORMATION_DEBUG
      for (Source *depsrc: sinks[i].sources) {
        deps.insert(depsrc->entry);
      }
    }
    for (Entry *dep: deps) {
      dep->evaluateTypes();
    }
    initInternals(sargs);
  }
}

/** @brief Evaluate output types based on input types via Entry::typefuns call, allocate memory. */
void Entry::updateTypes() {
  evaluateTypes();
}

/** @brief Update the transformation if it is not frozen and tainted. */
void Entry::touch() {
  if (tainted && !frozen) {
    update();
  }
}

/**
 * Returns i-th data. Does the calculation if needed.
 * @param i -- index of a Sink to read the data.
 * @return i-th Sink's Data.
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case input data is undefined.
 */
const Data<double> &Entry::data(int i) {
  if (i < 0 or static_cast<size_t>(i) > sinks.size()) {
    auto fmt = format("invalid sink idx %1%, have %2% sinks");
    throw CalculationError(this, (fmt % i % sinks.size()).str());
  }
  const Sink &sink = sinks[i];
  if (!sink.data) {
    auto fmt = format("sink %1% (%2%) have no type");
    throw CalculationError(this, (fmt % i % sink.name).str());
  }
  touch();
  return *sink.data;
}

void Entry::switchFunction(const std::string& name){
  auto it = functions.find(name);
  if(it==functions.end()){
    auto fmt = format("invalid function name %1%");
    throw std::runtime_error((fmt%name.data()).str());
  }
  fun = it->second.fun;

  funcname=name;
  evaluateTypes();
}

void Entry::initInternals(StorageTypesFunctionArgs& fargs){
  storages.clear();
  auto& itypes=fargs.ints;
  for (size_t i(0); i<itypes.size(); ++i) {
    auto& int_dtype=itypes[i];
    auto* storage = new Storage(this);
    TR_DPRINTF("stypes[%s, %i %s]: ", name.c_str(), static_cast<int>(i), storage->name.c_str());
#ifdef TRANSFORMATION_DEBUG
    int_dtype.dump();
#endif // TRANSFORMATION_DEBUG
    storage->data.reset(new Data<double>(int_dtype));
    storages.push_back(storage);
  }
}


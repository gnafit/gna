#include <set>
#include <iostream>

#include <boost/format.hpp>
using boost::format;

#include "Transformation.hh"
#include "Exceptions.hh"

using TransformationTypes::Source;
using TransformationTypes::Sink;

using TransformationTypes::SinkTypeError;
using TransformationTypes::SourceTypeError;
using TransformationTypes::CalculationError;

using TransformationTypes::InputHandle;
using TransformationTypes::OutputHandle;
using TransformationTypes::Handle;

using TransformationTypes::Args;
using TransformationTypes::Rets;
using TransformationTypes::Atypes;
using TransformationTypes::Rtypes;

using TransformationTypes::Entry;
using TransformationTypes::Base;

template <typename T>
std::string errorMessage(const std::string &type, const T *s,
                         const std::string &msg) {
  std::string name;
  if (s) {
    name = std::string(" ")+s->name;
  }
  std::string message = msg.empty() ? "unspecified error" : msg;
  return (boost::format("%2%%1%: %3%") % name % type % message).str();
}

/** @brief Constructor.
 *  @param s -- Sink with problematic type.
 *  @param message -- error message.
 */
SinkTypeError::SinkTypeError(const Sink *s, const std::string &message)
  : TypeError(errorMessage("sink", s, message)),
    sink(s)
{ }

/** @brief Constructor.
 *  @param s -- Source with problematic type.
 *  @param message -- error message.
 */
SourceTypeError::SourceTypeError(const Source *s, const std::string &message)
  : TypeError(errorMessage("source", s, message)),
    source(s)
{ }

/** @brief Constructor.
 *  @param e -- Entry where exception happens.
 *  @param message -- error message.
 */
CalculationError::CalculationError(const Entry *e, const std::string &message)
  : std::runtime_error(errorMessage("transformation", e, message)),
    entry(e)
{ }

/**
 * @brief Constructor.
 *
 * @param name -- Entry's name.
 * @param parent -- Base class instance to hold the Entry.
 */
Entry::Entry(const std::string &name, const Base *parent)
  : name(name), parent(parent), initializing(0), frozen(false)
{ }

/**
 * @brief Clone constructor.
 * @todo Cross check the description.
 *
 * @param other -- Entry to copy name, inputs and outputs from.
 * @param parent -- Base class instance to hold the Entry.
 */
Entry::Entry(const Entry &other, const Base *parent)
  : name(other.name), sources(other.sources.size()), sinks(other.sinks.size()),
    fun(), typefuns(), parent(parent), initializing(0), frozen(false)
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
 * @return OutputHandle for the new Sink.
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
  return fun(Args(this), Rets(this));
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
 * @brief Get vector of inputs.
 *
 * The method creates a new vector and fills it with InputHandle instances
 * for each Entry's Source.
 *
 * @return new vector of inputs.
 */
std::vector<InputHandle> Handle::inputs() const {
  std::vector<InputHandle> ret;
  auto &sources = m_entry->sources;
  std::transform(sources.begin(), sources.end(), std::back_inserter(ret),
                 [](Source &s) { return InputHandle(s); });
  return ret;
}

/**
 * @brief Get vector of outputs.
 *
 * The method creates a new vector and fills it with OutputHandle instances
 * for each Entry's Sink.
 *
 * @return new vector of outputs.
 */
std::vector<OutputHandle> Handle::outputs() const {
  std::vector<OutputHandle> ret;
  auto &sinks = m_entry->sinks;
  std::transform(sinks.begin(), sinks.end(), std::back_inserter(ret),
                 [](Sink &s) { return OutputHandle(s); });
  return ret;
}

/**
 * @brief Create a new input and connect to the SingleOutput transformation.
 *
 * New input name is copied from the output name.
 *
 * @param output -- SingleOutput transformation.
 * @return InputHandle for the new input.
 */
InputHandle Handle::input(SingleOutput &output) {
  OutputHandle outhandle = output.single();
  InputHandle inp = m_entry->addSource(outhandle.name());
  inp.connect(outhandle);
  return inp;
}

/**
 * @brief Create a new output with a same name as SingleOutput's output.
 *
 * @param out -- SingleOutput transformation.
 * @return OutputHandle for the new output.
 */
OutputHandle Handle::output(SingleOutput &out) {
  return output(out.single().name());
}

/**
 * @brief Print Entry's Sink and Source instances and their connection status.
 *
 * The data is printed to the stderr.
 */
void Handle::dumpObj() const {
  std::cerr << m_entry->name;
  std::cerr << std::endl;
  std::cerr << "    sources (" << m_entry->sources.size() << "):" << std::endl;
  int i = 0;
  for (Source &s: m_entry->sources) {
    std::cerr << "      " << i++ << ": " << s.name << ", ";
    if (s.sink) {
      std::cerr << "connected to ";
      std::cerr << s.sink->entry->name << "/" << s.sink->name << ", ";
      std::cerr << "type: ";
      s.sink->data->type.dump();
    } else {
      std::cerr << "not connected" << std::endl;
    }
  }
  std::cerr << "    sinks (" << m_entry->sinks.size() << "):" << std::endl;
  i = 0;
  for (Sink &s: m_entry->sinks) {
    std::cerr << "      " << i++ << ": " << s.name << ", ";
    std::cerr << s.sources.size() << " consumers";
    std::cerr << ", type: ";
    s.data->type.dump();
  }
}

Base::Base(const Base &other)
  : t_(*this), m_entries(other.m_entries.size())
{
  copyEntries(other);
}

Base &Base::operator=(const Base &other) {
  t_ = Accessor(*this);
  m_entries.reserve(other.m_entries.size());
  copyEntries(other);
  return *this;
}

void Base::copyEntries(const Base &other) {
  std::transform(other.m_entries.begin(), other.m_entries.end(),
                 std::back_inserter(m_entries),
                 [this](const Entry &e) { return new Entry{e, this}; });
}

/**
 * @brief Connect the Source to the Sink.
 *
 * After the connection:
 *   - the current Entry is subscribed to the Sink's taintflag to track dependencies.
 *   - Execute Sink::evaluateTypes() is called to define the input data types if not already defined.
 *   - Execute Entry::evaluateTypes() to check input data types and derive output data types if needed.
 *
 * @param newsink -- sink to connect to.
 *
 * @exception std::runtime_error if there already exists a connected Sink.
 */
void Source::connect(Sink *newsink) {
  if (sink) {
    std::cerr << this << " " << name << " " << sink->entry->name << "\n";
    throw std::runtime_error(
      (format("Transformation: source `%1%' is already connected to sink `%2%',"
              " won't connect to `%3%'") % name % sink->name % newsink->name)
       .str()
      );
  }
  //if (false) {
    //throw std::runtime_error("Transformation: connecting incompatible types");
  //}
  TR_DPRINTF("connecting source `%s'[%p] on `%s' to sink `%s'[%p] on `%s'\n", name.c_str(), (void*)this, entry->name.c_str(), newsink->name.c_str(), (void*)newsink, newsink->entry->name.c_str());
  sink = newsink;
  sink->entry->tainted.subscribe(entry->tainted);
  newsink->sources.push_back(this);
  try {
    newsink->entry->evaluateTypes();
    entry->evaluateTypes();
  } catch (const std::exception &exc) {
    std::cerr << "exception in types calculation: ";
    std::cerr << exc.what() << "\n";
    std::terminate();
  }
}

size_t Base::addEntry(Entry *e) {
  size_t idx = m_entries.size();
  m_entries.push_back(e);
  return idx;
}

Entry &Base::getEntry(const std::string &name) {
  for (Entry &e: m_entries) {
    if (e.name == name) {
      return e;
    }
  }
  throw KeyError(name, "transformation");
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
  Atypes args(this);
  Rtypes rets(this);
  bool success = false;
  TR_DPRINTF("evaluating types for %s: \n", name.c_str());
  try {
    for (auto &typefun: typefuns) {
      typefun(args, rets);
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
    for (size_t i = 0; i < sinks.size(); ++i) {
      if (!rets[i].buffer && sinks[i].data && sinks[i].data->type == rets[i]) {
        continue;
      }
      sinks[i].data.reset();
      if (rets[i].defined()) {
        sinks[i].data.reset(new Data<double>(rets[i], rets[i].buffer));
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

/**
 * @brief Assigns shape of each input to corresponding output.
 *
 * In case of single input and multiple outputs assign its size to each output.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception std::runtime_error in case the number of inputs and outputs is >1 and not the same.
 */
void Atypes::passAll(Atypes args, Rtypes rets) {
  if (args.size() == 1) {
    for (size_t i = 0; i < rets.size(); ++i) {
      rets[i] = args[0];
    }
  } else if (args.size() != rets.size()) {
    auto fmt = format("Transformation %1%: nargs != nrets");
    throw std::runtime_error((fmt % args.name()).str());
  } else {
    for (size_t i = 0; i < args.size(); ++i) {
      rets[i] = args[i];
    }
  }
}

/**
 * @brief Checks that all inputs are of the same type (shape and content description).
 *
 * Raises an exception otherwise.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case input types are not the same.
 */
void Atypes::ifSame(Atypes args, Rtypes rets) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i] != args[0]) {
      throw args.error(args[i], "inputs should have same type");
    }
  }
}

/**
 * @brief Checks that all inputs are of the same shape.
 *
 * Raises an exception otherwise.
 *
 * @param args -- source types.
 * @param rets -- output types.
 *
 * @exception SourceTypeError in case input shapes are not the same.
 */
void Atypes::ifSameShape(Atypes args, Rtypes rets) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i].shape != args[0].shape) {
      throw args.error(args[i], "inputs should have same shape");
    }
  }
}

/**
 * @brief Get i-th Sink DataType.
 *
 * @param i -- Sink index.
 * @return i-th Sink DataType.
 *
 * @exception std::runtime_error in case invalid index is queried.
 */
DataType &Rtypes::operator[](int i) {
  if (i < 0 || static_cast<size_t>(i) >= m_types->size()) {
    throw std::runtime_error(
      (format("invalid access to return type %1%, nsinks: %2%")
              % i % m_types->size()).str());
  }
  return (*m_types)[i];
}

/**
 * @brief Get i-th Source Data.
 * @param i -- index of a Source.
 * @return i-th Sources's Data as input (const).
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case input data is not initialized.
 */
const Data<double> &Args::operator[](int i) const {
  if (i < 0 or static_cast<size_t>(i) > m_entry->sources.size()) {
    auto fmt = format("invalid arg idx %1%, have %2% args");
    throw CalculationError(m_entry, (fmt % i % m_entry->sources.size()).str());
  }
  const Source &src = m_entry->sources[i];
  if (!src.materialized()) {
    auto fmt = format("arg %1% (%2%) have no type on evaluation");
    throw CalculationError(m_entry, (fmt % i % src.name).str());
  }
  src.sink->entry->touch();
  return *src.sink->data;
}

/**
 * @brief Get i-th Sink Data.
 * @param i -- index of a Sink.
 * @return i-th Sink's Data as output.
 *
 * @exception CalculationError in case invalid index is queried.
 * @exception CalculationError in case output data is not initialized.
 */
Data<double> &Rets::operator[](int i) const {
  if (i < 0 or static_cast<size_t>(i) > m_entry->sinks.size()) {
    auto fmt = format("invalid ret idx %1%, have %2% rets");
    throw CalculationError(m_entry, (fmt % i % m_entry->sinks.size()).str());
  }
  auto &data = m_entry->sinks[i].data;
  if (!data) {
    auto fmt = format("ret %1% (%2%) have no type on evaluation");
    throw CalculationError(m_entry, (fmt % i % m_entry->sinks[i].name).str());
  }

  return *m_entry->sinks[i].data;
}

/**
 * @brief Source type exception.
 * @param dt -- incorrect DataType.
 * @param message -- exception message.
 * @return exception.
 */
SourceTypeError Atypes::error(const DataType &dt, const std::string &message) {
  const Source *source = nullptr;
  for (size_t i = 0; i < m_entry->sources.size(); ++i) {
    if (&m_entry->sources[i].sink->data->type == &dt) {
      source = &m_entry->sources[i];
      break;
    }
  }
  return SourceTypeError(source, message);
}

/**
 * @brief Sink type exception.
 * @param dt -- incorrect DataType.
 * @param message -- exception message.
 * @return exception.
 */
SinkTypeError Rtypes::error(const DataType &dt, const std::string &message) {
  const Sink *sink = nullptr;
  for (size_t i = 0; i < m_types->size(); ++i) {
    if (&(*m_types)[i] == &dt) {
      sink = &m_entry->sinks[i];
      break;
    }
  }
  return SinkTypeError(sink, message);
}

/**
 * @brief Calculation error exception.
 * @param message -- exception message.
 * @return exception.
 */
CalculationError Rets::error(const std::string &message) {
  return CalculationError(this->m_entry, message);
}

/**
 * @brief Check the Entry.
 * @copydoc Entry::check()
 */
bool OutputHandle::check() const {
  return m_sink->entry->check();
}

/**
 * @brief Dump the Entry.
 * @copydoc Entry::dump()
 */
void OutputHandle::dump() const {
  return m_sink->entry->dump(0);
}

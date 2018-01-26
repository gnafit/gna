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

SinkTypeError::SinkTypeError(const Sink *s, const std::string &message)
  : TypeError(errorMessage("sink", s, message)),
    sink(s)
{ }

SourceTypeError::SourceTypeError(const Source *s, const std::string &message)
  : TypeError(errorMessage("source", s, message)),
    source(s)
{ }

CalculationError::CalculationError(const Entry *e, const std::string &message)
  : std::runtime_error(errorMessage("transformation", e, message)),
    entry(e)
{ }

Entry::Entry(const std::string &name, const Base *parent)
  : name(name), parent(parent), initializing(0), frozen(false)
{ }

Entry::Entry(const Entry &other, const Base *parent)
  : name(other.name), sources(other.sources.size()), sinks(other.sinks.size()),
    fun(), typefuns(), parent(parent), initializing(0), frozen(false)
{
  initSourcesSinks(other.sources, other.sinks);
}

template <typename InsT, typename OutsT>
void Entry::initSourcesSinks(const InsT &inputs, const OutsT &outputs) {
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(sources),
                 [this](const Source &s) { return new Source{s, this}; });
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(sinks),
                 [this](const Sink &s) { return new Sink{s, this}; });
}

InputHandle Entry::addSource(const std::string &name) {
  Source *s = new Source(name, this);
  sources.push_back(s);
  return InputHandle(*s);
}

OutputHandle Entry::addSink(const std::string &name) {
  Sink *s = new Sink(name, this);
  sinks.push_back(s);
  return OutputHandle(*s);
}

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

void Entry::evaluate() {
  fun(Args(this), Rets(this));
#ifdef GNA_CUDA_SUPPORT
  for(auto& sink : this->sinks){
    sink.data->setLocation( this->entryLoc );
  }
#endif
}

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

std::vector<InputHandle> Handle::inputs() const {
  std::vector<InputHandle> ret;
  auto &sources = m_entry->sources;
  std::transform(sources.begin(), sources.end(), std::back_inserter(ret),
                 [](Source &s) { return InputHandle(s); });
  return ret;
}

std::vector<OutputHandle> Handle::outputs() const {
  std::vector<OutputHandle> ret;
  auto &sinks = m_entry->sinks;
  std::transform(sinks.begin(), sinks.end(), std::back_inserter(ret),
                 [](Sink &s) { return OutputHandle(s); });
  return ret;
}

InputHandle Handle::input(SingleOutput &output) {
  OutputHandle outhandle = output.single();
  InputHandle inp = m_entry->addSource(outhandle.name());
  inp.connect(outhandle);
  
  return inp;
}

OutputHandle Handle::output(SingleOutput &out) {
  return output(out.single().name());
}

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

void Source::connect(Sink *newsink) {
  if (sink) {
    std::cerr << this << " " << name << " " << sink->entry->name << "\n";
    throw std::runtime_error(
      (format("Transformation: source `%1%' is already connected to sink `%2%',"
              " won't connect to `%3%'") % name % sink->name % newsink->name)
       .str()
      );
  }
  if (false) {
    throw std::runtime_error("Transformation: connecting incompatible types");
  }
  TR_DPRINTF("connecting source `%s'[%p] on `%s' to sink `%s'[%p] on `%s'\n", name.c_str(), (void*)this, entry->name.c_str(), newsink->name.c_str(), (void*)newsink, newsink->entry->name.c_str());
  sink = newsink;
  sink->entry->tainted.subscribe(entry->tainted);
  newsink->sources.push_back(this);
  try {
    sink->entry->evaluateTypes();
    entry->evaluateTypes();
  } catch (const std::exception &exc) {
    std::cerr << "exception in types calculation (bazzeg ROOT): ";
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

    // GPU: require GPU memory for previous transformation's sink
#ifdef GNA_CUDA_SUPPORT 
    if (entryLoc == Device) {  
      for (size_t i = 0; i < sources.size(); i++) {
	if ( sources[i].sink->entry->entryLoc != Device ) {
          sources[i].sink->data->require_gpu();
        }
      }
      for (auto &sink : sinks) {
        sink.data->require_gpu();
      }
    }
/*    if (entryLoc == Host) {
      for (auto& source : sources) {
        if ( source.sink->entry->entryLoc == Device) {
          source.sink->data->sync_D2H();
        }
      }
    }
*/
#endif
    for (Entry *dep: deps) {
      dep->evaluateTypes();
    }
  }
}

void Entry::updateTypes() {
  evaluateTypes();
}

void Entry::touch() {
  if (tainted && !frozen) {
    update();
  }
}

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
#ifdef GNA_CUDA_SUPPORT
  sink.data->sync( Host );
#endif
  return *sink.data;
}

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

void Atypes::ifSame(Atypes args, Rtypes rets) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i] != args[0]) {
      throw args.error(args[i], "inputs should have same type");
    }
  }
} 
void Atypes::ifSameShape(Atypes args, Rtypes rets) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i].shape != args[0].shape) {
      throw args.error(args[i], "inputs should have same shape");
    }
  }
}

DataType &Rtypes::operator[](int i) {
  if (i < 0 || static_cast<size_t>(i) >= m_types->size()) {
    throw std::runtime_error(
      (format("invalid access to return type %1%, nsinks: %2%")
              % i % m_types->size()).str());
  }
  return (*m_types)[i];
}

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
#ifdef GNA_CUDA_SUPPORT
  std::cout << "Sink loc = " << src.sink->data->dataLoc << std::endl;
  src.sink->data->sync(this->m_entry->entryLoc);
#endif
  return *src.sink->data;
}

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

SourceTypeError Atypes::error(const DataType &dt, const std::string &message) {
  const Source *source = nullptr;
  for (size_t i = 0; i < m_entry->sources.size(); ++i) {
    if (&m_entry->sources[i].sink->data->type == &dt) { source = &m_entry->sources[i]; break;
    }
  }
  return SourceTypeError(source, message);
}

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

bool OutputHandle::check() const {
  return m_sink->entry->check();
}

void OutputHandle::dump() const {
  return m_sink->entry->dump(0);
}

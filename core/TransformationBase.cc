#include <set>
#include <iostream>

#include <boost/format.hpp>
using boost::format;

#include "Transformation.hh"
#include "Exceptions.hh"

using TransformationTypes::Channel;
using TransformationTypes::Source;
using TransformationTypes::Sink;

using TransformationTypes::InputHandle;
using TransformationTypes::OutputHandle;
using TransformationTypes::Handle;

using TransformationTypes::Rets;
using TransformationTypes::Atypes;
using TransformationTypes::Rtypes;

using TransformationTypes::Entry;
using TransformationTypes::Base;

Entry::Entry(const std::string &name, const Base *parent)
  : name(name), parent(parent), initializing(0)
{
  taintedsrcs.subscribe(tainted);
}

Entry::Entry(const Entry &other, const Base *parent)
  : name(other.name), sources(other.sources.size()), sinks(other.sinks.size()),
    fun(), typefuns(), parent(parent), initializing(0)
{
  initSourcesSinks(other.sources, other.sinks);
}

template <typename InsT, typename OutsT>
void Entry::initSourcesSinks(const InsT &inputs, const OutsT &outputs) {
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(sources),
                 [this](const Channel &c) { return new Source{c, this}; });
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(sinks),
                 [this](const Channel &c) { return new Sink{c, this}; });
}

InputHandle Entry::addSource(const Channel &input) {
  Source *s = new Source(input, this);
  sources.push_back(s);
  return InputHandle(*s);
}

OutputHandle Entry::addSink(const Channel &output) {
  Sink *s = new Sink(output, this);
  sinks.push_back(s);
  return OutputHandle(*s);
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

void Handle::dump() const {
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
    throw std::runtime_error(
      (format("Transformation: source `%1' is already connected to sink `%2%,"
              "won't connect to `%3'") % name % sink->name % newsink->name)
       .str()
      );
  }
  if (!TransformationTypes::isCompatible(newsink, this)) {
    throw std::runtime_error("Transformation: connecting incompatible types");
  }
  TR_DPRINTF("connecting source `%s'[%p] on `%s' to sink `%s'[%p] on `%s'\n", name.c_str(), (void*)this, entry->name.c_str(), newsink->name.c_str(), (void*)newsink, newsink->entry->name.c_str());
  sink = newsink;
  sink->entry->tainted.subscribe(entry->taintedsrcs);
  newsink->sources.push_back(this);
  newsink->entry->evaluateTypes();
  entry->evaluateTypes();
}

bool TransformationTypes::isCompatible(const Channel *sink,
                                       const Channel *source) {
  if (source->channeltype.kind != sink->channeltype.kind) {
    return false;
  }
  return true;
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
  Status st = Status::Success;
  TR_DPRINTF("evaluating types for %s: \n", name.c_str());
  try {
    for (auto &typefun: typefuns) {
      typefun(args, rets);
    }
  } catch (Rtypes::Error) {
    st = Status::Failed;
  } catch (Atypes::Undefined) {
    st = Status::Undefined;
  }
  std::set<Entry*> deps;
  if (st == Status::Success) {
    TR_DPRINTF("types[%s]: success\n", name.c_str());
    for (size_t i = 0; i < sinks.size(); ++i) {
      if (sinks[i].data && sinks[i].data->type == rets[i]) {
        continue;
      }
      sinks[i].data.reset(new Data<double>(rets[i]));
      TR_DPRINTF("types[%s, %s]: ", name.c_str(), sinks[i].name.c_str());
#ifdef TRANSFORMATION_DEBUG
      sinks[i].data->type.dump();
#endif // TRANSFORMATION_DEBUG
      for (Source *depsrc: sinks[i].sources) {
        deps.insert(depsrc->entry);
      }
    }
  } else if (st == Status::Failed) {
    TR_DPRINTF("types[%s]: failed\n", name.c_str());
    throw std::runtime_error(
      (format("Transformation: type updates failed for `%1%'") % name).str()
      );
  } else if (st == Status::Undefined){
    TR_DPRINTF("types[%s]: undefined\n", name.c_str());
  }
  for (Entry *dep: deps) {
    dep->evaluateTypes();
  }
}

void Entry::updateTypes() {
  evaluateTypes();
}

void Atypes::passAll(Atypes args, Rtypes rets) {
  if (args.size() != rets.size()) {
    throw std::runtime_error("Transformation: nargs != nrets");
  }
  for (size_t i = 0; i < args.size(); ++i) {
    rets[i] = args[i];
  }
}

void Atypes::ifSame(Atypes args, Rtypes rets) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i] != args[0]) {
      throw rets.error(rets[0]);
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

Rets::Error Rets::error(const Data<double> &data) {
  const Sink *sink = nullptr;
  for (size_t i = 0; i < m_entry->sinks.size(); ++i) {
    if (m_entry->sinks[i].data.get() == &data) {
      sink = &m_entry->sinks[i];
    }
  }
  return Error(sink);
}

Rtypes::Error Rtypes::error(const DataType &dt) {
  const Sink *sink = nullptr;
  for (size_t i = 0; i < m_types->size(); ++i) {
    if (&(*m_types)[i] == &dt) {
      sink = &m_entry->sinks[i];
    }
  }
  return Error(sink);
}

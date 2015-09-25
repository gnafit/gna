#include <set>
#include <iostream>

#include <boost/format.hpp>
using boost::format;

#include "Transformation.hh"
using TransformationTypes::Channel;
using TransformationTypes::Source;
using TransformationTypes::Sink;

using TransformationTypes::InputHandle;
using TransformationTypes::OutputHandle;
using TransformationTypes::Handle;

using TransformationTypes::ArgumentTypes;

using TransformationTypes::Entry;
using TransformationTypes::Base;

using TransformationTypes::Function;
using TransformationTypes::TypesFunction;

Entry::Entry(const std::string &name,
             const std::vector<Channel> &inputs,
             const std::vector<Channel> &outputs,
             Function fun, TypesFunction typefun,
             const Base *parent)
  : name(name), sources(inputs.size()), sinks(outputs.size()),
    fun(fun), typefun(typefun), parent(parent)
{
  taintedsrcs.subscribe(tainted);
  initSourcesSinks(inputs, outputs);
}

Entry::Entry(const Entry &other, const Base *parent)
  : name(other.name), sources(other.sources.size()), sinks(other.sinks.size()),
    fun(), typefun(), parent(parent)
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
  updateTypes();
  return InputHandle(*s);
}

OutputHandle Entry::addSink(const Channel &output) {
  Sink *s = new Sink(output, this);
  sinks.push_back(s);
  updateTypes();
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
      s.sink->data.type.dump();
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
    s.data.type.dump();
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

TypesFunction Base::passTypesFunction(const std::string &name,
                                      const std::vector<Channel> &inputs,
                                      const std::vector<Channel> &outputs) {
  static auto passtypes = [](ArgumentTypes args, ReturnTypes rets) -> Status {
    if (rets.size() > 0) {
      for (size_t i = 0; i < args.size(); ++i) {
        rets[i] = args[i];
      }
    }
    return Status::Success;
  };
  if (outputs.size() > 0 && outputs.size() != inputs.size()) {
    throw std::runtime_error(
      (format("Transformation: nargs != nrets number for `%1%'") % name).str()
      );
  }
  return passtypes;
}

bool Source::connect(Sink *newsink) {
  if (sink) {
    throw std::runtime_error(
      (format("Transformation: source `%1' is already connected to sink `%2%,"
              "won't connect to `%3'") % name % sink->name % newsink->name)
       .str()
      );
  }
  if (!TransformationTypes::isCompatible(newsink, this)) {
    return false;
  }
  TR_DPRINTF("connecting source `%s'[%p] to sink `%s'[%p]", name.c_str(), this,
             newsink->name.c_str(), newsink);
  sink = newsink;
  sink->entry->tainted.subscribe(entry->taintedsrcs);
  newsink->sources.push_back(this);
  entry->evaluateTypes();
  return true;
}

bool TransformationTypes::isCompatible(const Channel *sink,
                                       const Channel *source) {
  if (source->channeltype.kind != sink->channeltype.kind) {
    return false;
  }
  return true;
}

size_t Base::addEntry(const std::string &name,
                      const std::vector<Channel> &inputs,
                      const std::vector<Channel> &outputs,
                      Function fun, TypesFunction typefun) {
  size_t idx = m_entries.size();
  Entry *e = new Entry{name, inputs, outputs, fun, typefun, this};
  m_entries.push_back(e);
  TR_DPRINTF("added entry `%s', idx = %lu", name.c_str(), idx);
#ifdef TRANSFORMATION_DEBUG
  for (const auto &s: m_entries.back().sources) {
    TR_DPRINTF("source `%s'", s.name.c_str());
  }
  for (const auto &s: m_entries.back().sinks) {
    TR_DPRINTF("sink `%s'", s.name.c_str());
    s.data.type.dump();
  }
#endif // TRANSFORMATION_DEBUG
  e->evaluateTypes();
  return idx;
}

Entry &Base::getEntry(const std::string &name) {
  for (Entry &e: m_entries) {
    if (e.name == name) {
      return e;
    }
  }
  throw std::runtime_error(
    (format("Transformation: can't find entry `%1%'") % name).str()
    );
}

void Entry::evaluateTypes() {
  ArgumentTypes args(this);
  ReturnTypes rets(this);
  Status st = typefun(args, rets);
  std::set<Entry*> deps;
  if (st == Status::Success) {
    for (size_t i = 0; i < sinks.size(); ++i) {
      if (sinks[i].data.type == rets[i]) {
        continue;
      }
      sinks[i].data.type = rets[i];
      sinks[i].data.allocate();
#ifdef TRANSFORMATION_DEBUG
      sinks[i].data.type.dump();
#endif // TRANSFORMATION_DEBUG
      for (Source *depsrc: sinks[i].sources) {
        deps.insert(depsrc->entry);
      }
    }
  } else if (st == Status::Failed) {
    throw std::runtime_error(
      (format("Transformation: type updates failed for `%1%'") % name).str()
      );
  }
  for (Entry *dep: deps) {
    dep->evaluateTypes();
  }
}

void Entry::updateTypes() {
  evaluateTypes();
}

template <typename Derived>
void Transformation<Derived>::rebindMemFunctions() {
  using namespace std::placeholders;
  size_t idx;
  MemFunction mfun;
  MemTypesFunction mtypefun;
  for (const auto &funs: m_memfuns) {
    std::tie(idx, mfun, mtypefun) = funs;
    if (mfun) {
      baseobj()->m_entries[idx].fun = std::bind(mfun, obj(), _1, _2);
    }
    if (mtypefun) {
      baseobj()->m_entries[idx].typefun = std::bind(mtypefun, obj(), _1, _2);
    }
  }
}

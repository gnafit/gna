#ifndef TRANSFORMATIONBASE_H
#define TRANSFORMATIONBASE_H

#include <string>
#include <vector>
#include <list>
#include <functional>
#include <type_traits>

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/noncopyable.hpp>

#include "Parameters.hh"
#include "Data.hh"

#ifdef TRANSFORMATION_DEBUG
#define TR_DPRINTF(...) do {                    \
  fprintf(stderr, __VA_ARGS__);                 \
  fprintf(stderr, "\n");                        \
} while (0)

#else
#define TR_DPRINTF(...)
#endif

template <typename T>
class Transformation;

namespace TransformationTypes {
  struct Channel {
    std::string name;
    DataType channeltype;
  };

  struct Entry;
  struct Source;
  struct Sink: public Channel, public boost::noncopyable {
    Sink(const Channel &chan, Entry *entry);

    std::unique_ptr<Data<double>> data;
    taintflag tainted;
    std::vector<Source*> sources;
    Entry *entry;
  };
  class OutputHandle;
  struct Source: public Channel, public boost::noncopyable {
    Source(const Channel &chan, Entry *entry)
      : Channel(chan), sink(nullptr), entry(entry)
      { }
    bool connect(Sink *newsink);
    bool materialized() const {
      return sink && sink->data && sink->data->defined();
    }
    const Sink *sink;
    Entry *entry;
  };

  class OutputHandle;
  class InputHandle {
    friend class OutputHandle;
  public:
    InputHandle(Source &source): m_source(&source) { }
    InputHandle(const InputHandle &other): InputHandle(*other.m_source) { }
    static InputHandle invalid(const std::string name);

    bool connect(const OutputHandle &out) const;

    const std::string &name() const { return m_source->name; }
  protected:
    TransformationTypes::Source *m_source;
  };

  class OutputHandle {
    friend class InputHandle;
  public:
    OutputHandle(Sink &sink): m_sink(&sink) { }
    OutputHandle(const OutputHandle &other): OutputHandle(*other.m_sink) { }
    static OutputHandle invalid(const std::string name);

    bool connect(const InputHandle &out) const;
    const TransformationTypes::Channel &channel() const { return *m_sink; }

    const std::string &name() const { return m_sink->name; }
  private:
    TransformationTypes::Sink *m_sink;
  };

  inline bool InputHandle::connect(const OutputHandle &out) const {
    return m_source->connect(out.m_sink);
  }

  inline bool OutputHandle::connect(const InputHandle &in) const {
    return in.m_source->connect(m_sink);
  }

  struct Args;
  struct Rets;
  struct Atypes;
  struct Rtypes;
  typedef std::function<Status(Args, Rets)> Function;
  typedef std::function<Status(Atypes, Rtypes)> TypesFunction;

  class Base;

  typedef boost::ptr_vector<Source> SourcesContainer;
  typedef boost::ptr_vector<Sink> SinksContainer;
  struct Entry: public boost::noncopyable {
    Entry(const std::string &name, const Base *parent);
    Entry(const Entry &other, const Base *parent);

    InputHandle addSource(const Channel &input);
    OutputHandle addSink(const Channel &output);

    Status evaluate();
    void update();
    void evaluateTypes();
    void updateTypes();

    const Data<double> &data(int i) {
      if (sinks[i].tainted) {
        update();
      }
      return *sinks[i].data;
    }

    std::string name;
    SourcesContainer sources;
    SinksContainer sinks;
    Function fun;
    TypesFunction typefun;
    taintflag tainted;
    taintflag taintedsrcs;
    const Base *parent;
  private:
    template <typename InsT, typename OutsT>
    void initSourcesSinks(const InsT &inputs, const OutsT &outputs);
  };

  class Handle {
  public:
    Handle(): m_entry(nullptr) { }
    Handle(Entry &entry) : m_entry(&entry) { }
    Handle(const Handle &other): Handle(*other.m_entry) { }

    const std::string &name() const { return m_entry->name; }
    std::vector<InputHandle> inputs() const;
    std::vector<OutputHandle> outputs() const;

    const Data<double> &operator[](int i) const { return m_entry->data(i); }

    void update(int i) const { (void)m_entry->data(i); }
    void updateTypes() { m_entry->updateTypes(); }

    taintflag tainted() { return m_entry->tainted; }

    void dump() const;
  protected:
    Entry *m_entry;
  };

  struct Args {
    Args(const Entry *e): m_entry(e) { }
    const Data<double> &operator[](int i) const {
      const Source &src = m_entry->sources[i];
      if (src.sink->tainted) {
        src.sink->entry->update();
      }
      return *src.sink->data;
    }
    size_t size() const { return m_entry->sources.size(); }
  private:
    const Entry *m_entry;
  };

  struct Rets {
  public:
    Rets(const Entry *e): m_entry(e) { }
    Data<double> &operator[](int i) const {
      return *m_entry->sinks[i].data;
    }
    size_t size() const { return m_entry->sinks.size(); }
  private:
    const Entry *m_entry;
  };

  struct Atypes {
    Atypes(const Entry *e): m_entry(e) { }
    const DataType &operator[](int i) const {
      if (!m_entry->sources[i].materialized()) {
        return DataType::undefined();
      }
      return m_entry->sources[i].sink->data->type;
    }
    size_t size() const { return m_entry->sources.size(); }
  private:
    const Entry *m_entry;
  };

  struct Rtypes {
  public:
    Rtypes(const Entry *e)
      : m_types(new std::vector<DataType>(e->sinks.size()))
      { }
    DataType &operator[](int i) {
      return (*m_types)[i];
    }
    size_t size() const { return m_types->size(); }
  protected:
    std::shared_ptr<std::vector<DataType> > m_types;
  };

  inline Sink::Sink(const Channel &chan, Entry *entry)
    : Channel(chan), entry(entry)
  {
    entry->tainted.subscribe(tainted);
  }

  inline Status Entry::evaluate() {
    return fun(Args(this), Rets(this));
  }

  inline void Entry::update() {
    Status status = evaluate();
    for (auto &sink: sinks) {
      sink.tainted = false;
      sink.data->status = status;
    }
    tainted = false;
  }

  class Accessor {
  public:
    Accessor() { }
    Accessor(Base &parent): m_parent(&parent) { }
    Handle operator[](int idx) const;
    Handle operator[](const std::string &name) const;
    size_t size() const;
  private:
    Base *m_parent;
  };

  class Base {
    template <typename T>
    friend class ::Transformation;
    template <typename T>
    friend class Initializer;
    friend class TransformationDescriptor;
    friend class Accessor;
  public:
    Base(const Base &other);
    Base &operator=(const Base &other);
  protected:
    Base(): t_(*this) { }
    bool connectChannel(Source &source, Base *sinkobj, Sink &sink);
    bool compatible(const Channel *sink, const Channel *source) const;
    Entry &getEntry(size_t idx) {
      return m_entries[idx];
    }
    Entry &getEntry(const std::string &name);
    InputHandle input_(const std::string &name, const Channel &input) {
      return getEntry(name).addSource(input);
    }
    OutputHandle output_(const std::string &name, const Channel &output) {
      return getEntry(name).addSink(output);
    }

    Accessor t_;
  private:
    size_t addEntry(const std::string &name);
    boost::ptr_vector<Entry> m_entries;
    void copyEntries(const Base &other);
  };

  inline Handle Accessor::operator[](int idx) const {
    return Handle(m_parent->getEntry(idx));
  }

  inline Handle Accessor::operator[](const std::string &name) const {
    TR_DPRINTF("accessing %s on %p", name.c_str(), m_parent);
    return Handle(m_parent->getEntry(name));
  }

  inline size_t Accessor::size() const {
    return m_parent->m_entries.size();
  }

  bool isCompatible(const Channel *sink, const Channel *source);

  template <typename T>
  class Initializer {
  public:
    typedef std::function<Status(T*,
                                 TransformationTypes::Args,
                                 TransformationTypes::Rets)> MemFunction;
    typedef std::function<Status(T*,
                                 TransformationTypes::Atypes,
                                 TransformationTypes::Rtypes)> MemTypesFunction;

    Initializer(Transformation<T> *obj, size_t idx)
      : m_obj(obj), m_idx(idx) { }
    Initializer<T> input(const std::string &name, const DataType &dt) {
      getEntry().addSource({name, dt});
      return *this;
    }
    Initializer<T> output(const std::string &name, const DataType &dt) {
      getEntry().addSink({name, dt});
      return *this;
    }
    Initializer<T> func(Function func) {
      m_obj->unbindMemFunction(m_idx);
      getEntry().fun = func;
      return *this;
    }
    Initializer<T> func(MemFunction func) {
      m_obj->bindMemFunction(m_idx, func);
      return *this;
    }
    Initializer<T> types(TypesFunction func) {
      m_obj->unbindMemTypesFunction(m_idx);
      getEntry().typefun = func;
      return *this;
    }
    Initializer<T> types(MemTypesFunction func) {
      m_obj->bindMemTypesFunction(m_idx, func);
      return *this;
    }
  protected:
    Entry &getEntry() {
      return m_obj->baseobj()->getEntry(m_idx);
    }
    Transformation<T> *m_obj;
    size_t m_idx;
  };
}

template <typename Derived>
class Transformation {
public:
  Transformation() { }
  Transformation(const Transformation<Derived> &other)
    : m_memFuncs(other.m_memFuncs), m_memTypesFuncs(other.m_memTypesFuncs)
  {
    rebindMemFunctions();
  }

  Transformation<Derived> &operator=(const Transformation<Derived> &other) {
    m_memFuncs = other.m_memFuncs;
    m_memTypesFuncs = other.m_memTypesFuncs;
    rebindMemFunctions();
    return *this;
  }
protected:
  friend class TransformationTypes::Initializer<Derived>;
  typedef typename TransformationTypes::Initializer<Derived> Initializer;
  typedef typename Initializer::MemFunction MemFunction;
  typedef typename Initializer::MemTypesFunction MemTypesFunction;

  TransformationTypes::Initializer<Derived>
  transformation_(const std::string &name) {
    auto idx = baseobj()->addEntry(name);
    return TransformationTypes::Initializer<Derived>(this, idx);
  }
private:
  Derived *obj() { return static_cast<Derived*>(this); }
  const Derived *obj() const { return static_cast<const Derived*>(this); }

  TransformationTypes::Base *baseobj() {
    return static_cast<TransformationTypes::Base*>(obj());
  }
  const TransformationTypes::Base *baseobj() const {
    return static_cast<const TransformationTypes::Base*>(obj());
  }

  std::list<std::tuple<size_t, MemFunction>> m_memFuncs;
  std::list<std::tuple<size_t, MemTypesFunction>> m_memTypesFuncs;

  void bindMemFunction(size_t idx, MemFunction);
  void bindMemTypesFunction(size_t idx, MemTypesFunction);
  void unbindMemFunction(size_t idx);
  void unbindMemTypesFunction(size_t idx);
  void rebindMemFunctions();
};

#endif // TRANSFORMATIONBASE_H

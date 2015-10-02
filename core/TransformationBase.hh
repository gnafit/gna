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

#define TRANSFORMATION_DEBUG

#ifdef TRANSFORMATION_DEBUG
#define TR_DPRINTF(...) do {                    \
  fprintf(stderr, __VA_ARGS__);                 \
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
    int initializing;
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
    InputHandle input(const Channel &input) {
      return m_entry->addSource(input);
    }
    OutputHandle output(const Channel &output) {
      return m_entry->addSink(output);
    }

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
    DataType &operator[](int i);
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

    Accessor t_;
  private:
    size_t addEntry(Entry *e);
    boost::ptr_vector<Entry> m_entries;
    void copyEntries(const Base &other);
  };

  inline Handle Accessor::operator[](int idx) const {
    return Handle(m_parent->getEntry(idx));
  }

  inline Handle Accessor::operator[](const std::string &name) const {
    TR_DPRINTF("accessing %s on %p\n", name.c_str(), (void*)m_parent);
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

    Initializer(Transformation<T> *obj, const std::string &name)
      : m_entry(new Entry(name, obj->baseobj())), m_obj(obj),
        m_nosubscribe(false)
    {
      m_entry->initializing++;
    }
    ~Initializer() {
      if (!m_entry) {
        return;
      }
      m_entry->initializing--;
      if (std::uncaught_exception()) {
        return;
      }
      if (m_entry->initializing == 0) {
        add();
      }
    }
    void add() {
      m_entry->initializing = 0;
      if (!m_nosubscribe) {
        m_obj->obj()->subscribe(m_entry->tainted);
      }
      size_t idx = m_obj->baseobj()->addEntry(m_entry);
      m_entry = nullptr;
      m_obj->addMemFunctions(idx, m_mfunc, m_mtfunc);
    }
    Initializer<T> input(const std::string &name, const DataType &dt) {
      m_entry->addSource({name, dt});
      return *this;
    }
    Initializer<T> output(const std::string &name, const DataType &dt) {
      m_entry->addSink({name, dt});
      return *this;
    }
    Initializer<T> func(Function func) {
      m_mfunc = nullptr;
      m_entry->fun = func;
      return *this;
    }
    Initializer<T> func(MemFunction func) {
      using namespace std::placeholders;
      m_mfunc = func;
      m_entry->fun = std::bind(func, m_obj->obj(), _1, _2);
      return *this;
    }
    Initializer<T> types(TypesFunction func) {
      m_mtfunc = nullptr;
      m_entry->typefun = func;
      return *this;
    }
    Initializer<T> types(MemTypesFunction func) {
      using namespace std::placeholders;
      m_mtfunc = func;
      m_entry->typefun = std::bind(func, m_obj->obj(), _1, _2);
      return *this;
    }
    Initializer<T> dont_subscribe() {
      m_nosubscribe = true;
      return *this;
    }
  protected:
    Entry *m_entry;
    Transformation<T> *m_obj;

    MemFunction m_mfunc;
    MemTypesFunction m_mtfunc;

    bool m_nosubscribe;
  };
}

template <typename Derived>
class Transformation {
public:
  Transformation() { }
  Transformation(const Transformation<Derived> &other)
    : m_memFuncs(other.m_memFuncs)
  {
    rebindMemFunctions();
  }

  Transformation<Derived> &operator=(const Transformation<Derived> &other) {
    m_memFuncs = other.m_memFuncs;
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
    return TransformationTypes::Initializer<Derived>(this, name);
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

  std::list<std::tuple<size_t, MemFunction, MemTypesFunction>> m_memFuncs;

  void addMemFunctions(size_t idx, MemFunction func, MemTypesFunction tfunc) {
    if (func || tfunc) {
      m_memFuncs.emplace_back(idx, func, tfunc);
    }
  }
  void rebindMemFunctions() {
    using namespace std::placeholders;
    auto &entries = baseobj()->m_entries;
    for (const auto &f: m_memFuncs) {
      auto idx = std::get<0>(f);
      if (std::get<1>(f)) {
        entries[idx].fun = std::bind(std::get<1>(f), obj(), _1, _2);
      }
      if (std::get<2>(f)) {
        entries[idx].typefun = std::bind(std::get<2>(f), obj(), _1, _2);
      }
    }
  }
};

#endif // TRANSFORMATIONBASE_H

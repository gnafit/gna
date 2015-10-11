#ifndef TRANSFORMATIONBASE_H
#define TRANSFORMATIONBASE_H

#include <string>
#include <vector>
#include <list>
#include <functional>
#include <type_traits>

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/noncopyable.hpp>
#include <boost/optional.hpp>

#include "Parameters.hh"
#include "Data.hh"

// #define TRANSFORMATION_DEBUG

#ifdef TRANSFORMATION_DEBUG
#define TR_DPRINTF(...) do {                    \
  fprintf(stderr, __VA_ARGS__);                 \
} while (0)

#else
#define TR_DPRINTF(...)
#endif

template <typename T>
class Transformation;

class GNAObject;
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
    void connect(Sink *newsink);
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

    void connect(const OutputHandle &out) const;

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

    const TransformationTypes::Channel &channel() const { return *m_sink; }

    const std::string &name() const { return m_sink->name; }
  private:
    TransformationTypes::Sink *m_sink;
  };

  inline void InputHandle::connect(const OutputHandle &out) const {
    return m_source->connect(out.m_sink);
  }

  struct Args;
  struct Rets;
  struct Atypes;
  struct Rtypes;
  typedef std::function<void(Args, Rets)> Function;
  typedef std::function<void(Atypes, Rtypes)> TypesFunction;

  class Base;

  typedef boost::ptr_vector<Source> SourcesContainer;
  typedef boost::ptr_vector<Sink> SinksContainer;
  struct Entry: public boost::noncopyable {
    Entry(const std::string &name, const Base *parent);
    Entry(const Entry &other, const Base *parent);

    InputHandle addSource(const Channel &input);
    OutputHandle addSink(const Channel &output);

    void evaluate();
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
    std::vector<TypesFunction> typefuns;
    taintflag tainted;
    taintflag taintedsrcs;
    const Base *parent;
    int initializing;
  private:
    template <typename InsT, typename OutsT>
    void initSourcesSinks(const InsT &inputs, const OutsT &outputs);
  };
  typedef boost::ptr_vector<Entry> Container;
  
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
    class Error {
    public:
      Error(const Sink *s) : sink(s) { }
      const Sink *sink;
    };
    Rets(const Entry *e): m_entry(e) { }
    Data<double> &operator[](int i) const {
      return *m_entry->sinks[i].data;
    }
    size_t size() const { return m_entry->sinks.size(); }
    Error error(const Data<double> &data);
  private:
    const Entry *m_entry;
  };

  struct Atypes {
    class Undefined {
    public:
      Undefined(const Source *s) : source(s) { }
      const Source *source;
    };
    Atypes(const Entry *e): m_entry(e) { }
    const DataType &operator[](int i) const {
      if (!m_entry->sources[i].materialized()) {
        throw Undefined(&m_entry->sources[i]);
      }
      return m_entry->sources[i].sink->data->type;
    }
    size_t size() const { return m_entry->sources.size(); }

    static void passAll(Atypes args, Rtypes rets);
    template <size_t Arg, size_t Ret = Arg>
    static void pass(Atypes args, Rtypes rets);
    static void ifSame(Atypes args, Rtypes rets);
  private:
    const Entry *m_entry;
  };

  struct Rtypes {
  public:
    class Error {
    public:
      Error(const Sink *s) : sink(s) { }
      const Sink *sink;
    };
    Rtypes(const Entry *e)
      : m_entry(e), m_types(new std::vector<DataType>(e->sinks.size()))
      { }
    DataType &operator[](int i);
    size_t size() const { return m_types->size(); }
    Error error(const DataType &dt);
  protected:
    const Entry *m_entry;
    std::shared_ptr<std::vector<DataType> > m_types;
  };

  template <size_t Arg, size_t Ret>
  inline void Atypes::pass(Atypes args, Rtypes rets) {
    if (Arg >= args.size()) {
      throw std::runtime_error("Transformation: invalid Arg index");
    }
    if (Ret >= rets.size()) {
      throw std::runtime_error("Transformation: invalid Ret index");
    }
    rets[Ret] = args[Arg];
  }

  inline Sink::Sink(const Channel &chan, Entry *entry)
    : Channel(chan), entry(entry)
  {
    entry->tainted.subscribe(tainted);
  }

  inline void Entry::evaluate() {
    return fun(Args(this), Rets(this));
  }

  inline void Entry::update() {
    Status status = Status::Success;
    try {
      evaluate();
    } catch (Rets::Error) {
      status = Status::Failed;
    }
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
    friend class ::GNAObject;
  public:
    Base(const Base &other);
    Base &operator=(const Base &other);
  protected:
    Base(): t_(*this) { }
    Base(size_t maxentries): Base() {
      m_maxEntries = maxentries;
    }
    void connectChannel(Source &source, Base *sinkobj, Sink &sink);
    bool compatible(const Channel *sink, const Channel *source) const;
    Entry &getEntry(size_t idx) {
      return m_entries[idx];
    }
    Entry &getEntry(const std::string &name);

    Accessor t_;
  private:
    size_t addEntry(Entry *e);
    boost::ptr_vector<Entry> m_entries;
    boost::optional<size_t> m_maxEntries;
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
    typedef std::function<void(T*,
                               TransformationTypes::Args,
                               TransformationTypes::Rets)> MemFunction;
    typedef std::function<void(T*,
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
        delete m_entry;
        return;
      }
      if (m_entry->initializing == 0) {
        add();
      }
    }
    void add() {
      auto *baseobj = m_obj->baseobj();
      if (baseobj->m_maxEntries &&
          baseobj->m_entries.size()+1 > baseobj->m_maxEntries) {
        throw std::runtime_error("too much transformations");
      }
      if (m_entry->typefuns.empty()) {
        m_entry->typefuns.push_back(Atypes::passAll);
      }
      m_entry->initializing = 0;
      if (!m_nosubscribe) {
        m_obj->obj()->subscribe(m_entry->tainted);
      }
      size_t idx = baseobj->addEntry(m_entry);
      m_entry = nullptr;
      if (m_mfunc) {
        m_obj->addMemFunction(idx, m_mfunc);
      }
      for (const auto &f: m_mtfuncs) {
        m_obj->addMemTypesFunction(idx, std::get<0>(f), std::get<1>(f));
      }
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
      m_entry->typefuns.push_back(func);
      return *this;
    }
    Initializer<T> types(MemTypesFunction func) {
      using namespace std::placeholders;
      m_mtfuncs.emplace_back(m_entry->typefuns.size(), func);
      m_entry->typefuns.push_back(std::bind(func, m_obj->obj(), _1, _2));
      return *this;
    }
    template <typename FuncA, typename FuncB>
    Initializer<T> types(FuncA func1, FuncB func2) {
      types(func1);
      types(func2);
      return *this;
    }
    template <typename FuncA, typename FuncB, typename FuncC>
    Initializer<T> types(FuncA func1, FuncB func2, FuncC func3) {
      types(func1);
      types(func2);
      types(func3);
      return *this;
    }
    template <typename Changeable>
    Initializer<T> depends(Changeable v) {
      v.subscribe(m_entry->tainted);
      m_nosubscribe = true;
      return *this;
    }
    template <typename Changeable, typename... Rest>
    Initializer<T> depends(Changeable v, Rest... rest) {
      depends(v);
      return depends(rest...);
    }
    Initializer<T> dont_subscribe() {
      m_nosubscribe = true;
      return *this;
    }
  protected:
    Entry *m_entry;
    Transformation<T> *m_obj;

    MemFunction m_mfunc;
    std::vector<std::tuple<size_t, MemTypesFunction>> m_mtfuncs;

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

  std::list<std::tuple<size_t, MemFunction>> m_memFuncs;
  std::list<std::tuple<size_t, size_t, MemTypesFunction>> m_memTypesFuncs;

  void addMemFunction(size_t idx, MemFunction func) {
    m_memFuncs.emplace_back(idx, func);
  }
  void addMemTypesFunction(size_t idx, size_t fidx, MemTypesFunction func) {
    m_memTypesFuncs.emplace_back(idx, fidx, func);
  }
  void rebindMemFunctions() {
    using namespace std::placeholders;
    auto &entries = baseobj()->m_entries;
    for (const auto &f: m_memFuncs) {
      auto idx = std::get<0>(f);
      entries[idx].fun = std::bind(std::get<1>(f), obj(), _1, _2);
    }
    for (const auto &f: m_memTypesFuncs) {
      auto idx = std::get<0>(f);
      auto fidx = std::get<1>(f);
      entries[idx].typefuns[fidx] = std::bind(std::get<2>(f), obj(), _1, _2);
    }
  }
};

#endif // TRANSFORMATIONBASE_H

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
class SingleOutput;
namespace TransformationTypes {
  struct Entry;
  struct Source;

  /**
   * @brief Definition of a single transformation output (Sink).
   *
   * Sink instance carries the actual Data.
   *
   * It also knows where this data is connected to (Sink::sources).
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Sink: public boost::noncopyable {
    Sink(const std::string &name, Entry *entry)
      : name(name), entry(entry) { }            ///< Constructor.
    Sink(const Sink &other, Entry *entry)
      : name(other.name), entry(entry) { }      ///< Copy constructor.

    std::string name;                           ///< Sink's name.
    std::unique_ptr<Data<double>> data;         ///< Sink's Data.
    std::vector<Source*> sources;               ///< Container with Source pointers which use this Sink as their input.
    Entry *entry;                               ///< Pointer to the transformation Entry this Sink belongs to.
  };

  class OutputHandle;

  /**
   * @brief Definition of a single transformation input (Source).
   *
   * Source instance is a link to the other transformation Entry's Sink,
   * that carries the transformation output.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Source: public boost::noncopyable {
    Source(const std::string &name, Entry *entry)
      : name(name), entry(entry) { }               ///< Constructor.
    Source(const Source &other, Entry *entry)
      : name(other.name), entry(entry) { }         ///< Copy constructor.

    void connect(Sink *newsink);                   ///< Connect the Source to the Sink.

    /**
     * @brief Check if the input data is allocated.
     * @return true if input data is allocated.
     */
    bool materialized() const {
      return sink && sink->data;
    }
    std::string name;                             ///< Source's name.
    const Sink *sink = nullptr;                   ///< Pointer to the Sink the Source is connected to.
    Entry *entry;                                 ///< Entry pointer the Source belongs to.
  };

  class TypeError: public std::runtime_error {
  public:
    TypeError(const std::string &message)
      : std::runtime_error(message) { }
  };

  class SinkTypeError: public TypeError {
  public:
    SinkTypeError(const Sink *s, const std::string &message);

    const Sink *sink;
  };

  class SourceTypeError: public TypeError {
  public:
    SourceTypeError(const Source *s, const std::string &message);

    const Source *source;
  };

  class CalculationError: public std::runtime_error {
  public:
    CalculationError(const Entry *e, const std::string &message);

    const Entry *entry;
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

    const void *rawptr() const { return static_cast<const void*>(m_source); }
    const size_t hash() const { return reinterpret_cast<size_t>(rawptr()); }
  protected:
    TransformationTypes::Source *m_source;
  };

  class OutputHandle {
    friend class InputHandle;
  public:
    OutputHandle(Sink &sink): m_sink(&sink) { }
    OutputHandle(const OutputHandle &other): OutputHandle(*other.m_sink) { }
    static OutputHandle invalid(const std::string name);

    const std::string &name() const { return m_sink->name; }

    bool check() const;
    void dump() const;

    const double *data() const;
    const double *view() const { return m_sink->data->x.data(); }
    const DataType &datatype() const { return m_sink->data->type; }

    const void *rawptr() const { return static_cast<const void*>(m_sink); }
    const size_t hash() const { return reinterpret_cast<size_t>(rawptr()); }
    bool depends(changeable x) const;
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
  /**
   * @brief Function, that does the actual calculation.
   *
   * This function is used to define the transformation via Entry::fun
   * and is executed via Entry::update() or Entry::touch().
   *
   * @param args -- container with transformation inputs (Args).
   * @param rets -- container with transformation outputs (Args).
   */
  typedef std::function<void(Args, Rets)> Function;

  /**
   * @brief Function, that does the input types checking and output types derivation.
   *
   * The function is used within Entry::evaluateTypes() and Entry::updateTypes().
   *
   * @param atypes -- container with transformation inputs' types (Atypes).
   * @param rtypes -- container with transformation outputs' types (Rtypes).
   */
  typedef std::function<void(Atypes, Rtypes)> TypesFunction;

  class Base;

  typedef boost::ptr_vector<Source> SourcesContainer;   ///< Container for Source instances.
  typedef boost::ptr_vector<Sink> SinksContainer;       ///< Container for Sink instances.
  /**
   * @brief Definition of a single transformation.
   *
   * Entry defines a transformation that:
   *   - has zero or more inputs: Source instances.
   *   - has one or more outputs: Sink instances.
   *   - has a function Entry::fun that defines the transformation.
   *   - may have several type functions (Entry::typefuns), that check the input types
   *     and derive the output types.
   *
   * Entry has a taintflag (Entry::taintflag), then defines whether the Entry's Sink instances
   * contain up to date output data.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Entry: public boost::noncopyable {
    Entry(const std::string &name, const Base *parent); ///< Constructor.
    Entry(const Entry &other, const Base *parent);      ///< Copy constructor.

    InputHandle addSource(const std::string &name);     ///< Initialize and return new Source.
    OutputHandle addSink(const std::string &name);      ///< Initialize and return new Sink.

    void evaluate();                                    ///< Do actual calculation by calling Entry::fun.
    void update();                                      ///< Do actual calculation by calling Entry::fun via evaluate() and resets the taintflag.
    void evaluateTypes();                               ///< Evaluate output types based on input types via Entry::typefuns call, allocate memory.
    void updateTypes();                                 ///< Evaluate output types based on input types via Entry::typefuns call, allocate memory.

    void touch();                                       ///< Update the transformation if it is not frozen and tainted.
    const Data<double> &data(int i);                    ///< Evaluates the function if needed and returns i-th data.

    void freeze() { frozen = true; }                    ///< Freeze the Entry. While entry is frozen the taintflag is not propagated. Entry is always up to date.
    void unfreeze() { frozen = false; }                 ///< Unfreeze the Entry. Enables the taintflag propagation.

    bool check() const;                                 ///< Checks that Data are initialized.
    void dump(size_t level = 0) const;                  ///< Recursively print Source names and their connection status.

    std::string name;                                   ///< Transformation name
    SourcesContainer sources;                           ///< Transformation inputs (sources)
    SinksContainer sinks;                               ///< Transformation outputs (sinks)
    Function fun;                                       ///< The function that does actual calculation
    std::vector<TypesFunction> typefuns;                ///< Vector of TypeFunction instances
    taintflag tainted;                                  ///< taintflag shows whether the result is up to date
    const Base *parent;                                 ///< Base class, containing the transformation Entry.
    int initializing;                                   ///< Initialization status. initializing>0 when Entry is being configured via Initializer.
    bool frozen;                                        ///< If Entry is frozen, it is not updated even if tainted.
    bool usable;                                        ///< Unused.

  private:
    template <typename InsT, typename OutsT>
    void initSourcesSinks(const InsT &inputs, const OutsT &outputs); ///< Initialize the clones for inputs and outputs.
  };
  typedef boost::ptr_vector<Entry> Container; ///< Container for Entry pointers.

  inline const double *OutputHandle::data() const {
    m_sink->entry->touch();
    return view();
  }

  inline bool OutputHandle::depends(changeable x) const {
    return m_sink->entry->tainted.depends(x);
  }

  class Handle {
  public:
    Handle(): m_entry(nullptr) { }
    Handle(Entry &entry) : m_entry(&entry) { }
    Handle(const Handle &other): Handle(*other.m_entry) { }

    const std::string &name() const { return m_entry->name; }
    std::vector<InputHandle> inputs() const;
    std::vector<OutputHandle> outputs() const;
    InputHandle input(const std::string &name) {
      return m_entry->addSource(name);
    }
    InputHandle input(SingleOutput &output);
    OutputHandle output(const std::string &name) {
      return m_entry->addSink(name);
    }
    OutputHandle output(SingleOutput &output);

    const Data<double> &operator[](int i) const { return m_entry->data(i); }

    void update(int i) const { (void)m_entry->data(i); }
    void updateTypes() { m_entry->updateTypes(); }

    void unfreeze() { m_entry->frozen = false; }

    void taint() { m_entry->tainted.taint(); }
    taintflag tainted() { return m_entry->tainted; }

    bool check() const { return m_entry->check(); }
    void dump() const { m_entry->dump(0); }
    void dumpObj() const;
  protected:
    Entry *m_entry;
  };

  class OpenHandle : public Handle {
  public:
      OpenHandle(const Handle& other) : Handle(other){};
      Entry* getEntry() { return m_entry; }
  };


  /**
   * @brief Access the transformation inputs.
   *
   * Args instance is passed to the Entry::fun function and is used to retrieve input data for the transformation.
   *
   * Args gives read-only access to the Source instances through Entry instance.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Args {
  public:
    /**
     * @brief Args constructor.
     * @param e -- Entry instance. Args will get access to Entry's sources.
     */
    Args(const Entry *e): m_entry(e) { }

    /**
     * @brief Get i-th Source Data.
     * @param i -- index of a Source.
     * @return i-th Sources's Data as input (const).
     */
    const Data<double> &operator[](int i) const;

    /**
     * @brief Get number of transformation sources.
     * @return Number of transformation Source instances.
     */
    size_t size() const { return m_entry->sources.size(); }
  private:
    const Entry *m_entry; ///< Entry instance to access Sources.
  };

  /**
   * @brief Access the transformation outputs.
   *
   * Rets instance is passed to the Entry::fun function and is used to write output data of the transformation.
   *
   * Rets gives write access to the Sink instances through Entry instance.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Rets {
  public:
    /**
     * @brief Rets constructor.
     * @param e -- Entry instance. Rets will get access to Entry's sinks.
     */
    Rets(Entry *e): m_entry(e) { }

    /**
     * @brief Get i-th Sink Data.
     * @param i -- index of a Sink.
     * @return i-th Sink's Data as output.
     */
    Data<double> &operator[](int i) const;

    /**
     * @brief Get number of transformation sinks.
     * @return Number of transformation Sink instances.
     */
    size_t size() const { return m_entry->sinks.size(); }

    /**
     * @brief Calculation error exception.
     * @param message -- exception message.
     * @return exception.
     */
    CalculationError error(const std::string &message = "");

    /**
     * @brief Freeze the Entry.
     *
     * While entry is frozen the taintflag is not propagated. Entry is always up to date.
     */
    void freeze()  { m_entry->freeze(); }

    /**
     * @brief Unfreeze the Entry.
     *
     * Enables the taintflag propagation.
     */
    void unfreeze()  { m_entry->unfreeze(); }

  private:
    Entry *m_entry; ///< Entry instance to access Sinks.
  };

  /**
   * @brief Access the transformation inputs' DataType (read only).
   *
   * It's needed to:
   *   - check the consistency of the inputs in the run time.
   *   - derive the output DataType instances.
   *
   * Atypes instance is passed to each of the Entry's TypeFunction instances.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Atypes {
    /**
     * @brief An exception for uninitialized Source instance
     */
    class Undefined {
    public:
      Undefined(const Source *s = nullptr) : source(s) { }
      const Source *source;
    };
    /**
     * @brief Atypes constructor.
     * @param e -- Entry instance. Atypes will get access to Entry's source types.
     */
    Atypes(const Entry *e): m_entry(e) { }

    /**
     * @brief Direct access to Sink instance, which is used as Source for the transformation.
     *
     * @param i -- Source number to return its Sink.
     * @return i-th Source's Sink instance.
     *
     * @exception Undefined in case input data is not initialized.
     */
    const Sink *sink(int i) const {
      if (!m_entry->sources[i].materialized()) {
        throw Undefined(&m_entry->sources[i]);
      }
      return m_entry->sources[i].sink;
    }

    /**
     * @brief Get i-th Source DataType (const).
     * @param i -- Source index.
     * @return i-th Source DataType.
     */
    const DataType &operator[](int i) const {
      return sink(i)->data->type;
    }

    /**
     * @brief Get number of Source instances.
     * @return number of sources.
     */
    size_t size() const { return m_entry->sources.size(); }

    static void passAll(Atypes args, Rtypes rets); ///< Assigns shape of each input to corresponding output.

    template <size_t Arg, size_t Ret = Arg>
    static void pass(Atypes args, Rtypes rets);    ///< Assigns shape of Arg-th input to Ret-th output.

    static void ifSame(Atypes args, Rtypes rets);  ///< Checks that all inputs are of the same type (shape and content description).
    static void ifSameShape(Atypes args, Rtypes rets); ///< Checks that all inputs are of the same shape.

    template <size_t Arg>
    static void ifHist(Atypes args, Rtypes rets);  ///< Checks if Arg-th input is a histogram (DataKind=Histogram).

    template <size_t Arg>
    static void ifPoints(Atypes args, Rtypes rets); ///< Checks if Arg-th input is an array (DataKind=Points).

    /**
     * @brief Source type exception.
     * @param dt -- incorrect DataType.
     * @param message -- exception message.
     * @return exception.
     */
    SourceTypeError error(const DataType &dt, const std::string &message = "");

    /**
     * @brief Get Entry's name
     * @return Entry's name
     */
    const std::string &name() const { return m_entry->name; }

    /**
     * @brief Empty Undefined exception.
     * @return Empty Undefined exception.
     */
    Undefined undefined() { return Undefined(); }
  private:
    const Entry *m_entry; ///< Entry instance to access Source DataType.
  };

  /**
   * @brief Storage for the new transformation's outputs' DataType types.
   *
   * It's needed to store the derived outputs' DataType types.
   *
   * Rtypes instance is passed to each of the Entry's TypeFunction functions.
   *
   * @note Rtypes will NOT write to Entry's output DataType types by itself. The actual assignment happens in the Entry::evaluateTypes() method.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Rtypes {
  public:
    /**
     * @brief Rtypes constructor.
     *
     * Rtypes will NOT write to Entry's output DataType types by itself.
     *
     * @param e -- Entry instance.
     */
    Rtypes(const Entry *e)
      : m_entry(e), m_types(new std::vector<DataType>(e->sinks.size()))
      { }

    /**
     * @brief Get i-th Sink DataType.
     * @param i -- Sink index.
     * @return i-th Sink DataType.
     */
    DataType &operator[](int i);

    /**
     * @brief Get number of Sink instances.
     * @return number of sinks.
     */
    size_t size() const { return m_types->size(); }

    /**
     * @brief Sink type exception.
     * @param dt -- incorrect DataType.
     * @param message -- exception message.
     * @return exception.
     */
    SinkTypeError error(const DataType &dt, const std::string &message = "");

    /**
     * @brief Get Entry's name
     * @return Entry's name
     */
    const std::string &name() const { return m_entry->name; }

  protected:
    const Entry *m_entry; ///< Entry instance.
    std::shared_ptr<std::vector<DataType> > m_types; ///< Storage for the output DataType types.
  };

  /**
   * @brief Assigns shape of Arg-th input to Ret-th output
   *
   * @tparam Arg -- index of Arg to read the type.
   * @tparam Ret -- index of Ret to write the type (by default Ret=Arg)
   *
   * @param args -- source types.
   * @param rets -- output types.
   *
   * @exception std::runtime_error in case of invalid index is passed.
   */
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

  /**
   * @brief Checks if Arg-th input is a histogram (DataKind=Histogram).
   *
   * Raises an exception otherwise.
   *
   *  @tparam Arg -- index of Arg to check.
   *
   *  @param args -- source types.
   *  @param rets -- output types.
   *
   *  @exception std::runtime_error in case input data is not a histogram.
   */
  template <size_t Arg>
  inline void Atypes::ifHist(Atypes args, Rtypes rets) {
    if (args[Arg].kind!=DataKind::Hist) {
      throw std::runtime_error("Transformation: Arg should be a histogram");
    }
  }

  /**
   * @brief Checks if Arg-th input is an array (DataKind=Points).
   *
   * Raises an exception otherwise.
   *
   * @tparam Arg -- index of Arg to check.
   *
   * @param args -- source types.
   * @param rets -- output types.
   *
   *  @exception std::runtime_error in case input data is not an array.
   */
  template <size_t Arg>
  inline void Atypes::ifPoints(Atypes args, Rtypes rets) {
    if (args[Arg].kind!=DataKind::Points) {
      throw std::runtime_error("Transformation: Arg should be an array");
    }
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

  template <typename T>
  class Initializer;
  class Base: public boost::noncopyable {
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
    void connect(Source &source, Base *sinkobj, Sink &sink);
    Entry &getEntry(size_t idx) {
      return m_entries[idx];
    }
    Entry &getEntry(const std::string &name);

    template <typename T>
    Initializer<T> transformation_(T *obj, const std::string &name) {
      return Initializer<T>(obj, name);
    }

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

    Initializer<T> input(const std::string &name) {
      m_entry->addSource(name);
      return *this;
    }

    Initializer<T> output(const std::string &name) {
      m_entry->addSink(name);
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

    Initializer<T> finalize() {
      m_entry->evaluateTypes();
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

private:
  friend class TransformationTypes::Initializer<Derived>;
  typedef typename TransformationTypes::Initializer<Derived> Initializer;
  typedef typename Initializer::MemFunction MemFunction;
  typedef typename Initializer::MemTypesFunction MemTypesFunction;
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

class SingleOutput {
public:
  virtual ~SingleOutput() { }
  virtual TransformationTypes::OutputHandle single() = 0;
  const double *data() { return single().data(); }
  const double *view() { return single().view(); }
  const DataType &datatype() { return single().datatype(); }
};

#endif // TRANSFORMATIONBASE_H

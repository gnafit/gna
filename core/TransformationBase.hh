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
/**
 * @brief A namespace for transformations.
 * The namespace defines Entry, Sink, Source and Base classes, necessary to deal
 * with transformations. Helper classes are also provided here.
 * @author Dmitry Taychenachev
 * @date 2015
 */
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
    /**
     * @brief Constructor.
     * @param name -- Sink name.
     * @param entry -- Entry pointer Sink belongs to.
     */
    Sink(const std::string &name, Entry *entry)
      : name(name), entry(entry) { }
    /**
     * @brief Clone constructor.
     * @param name -- other Sink to get the name from.
     * @param entry -- Entry pointer Sink belongs to.
     */
    Sink(const Sink &other, Entry *entry)
      : name(other.name), entry(entry) { }

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
    /**
     * @brief Constructor.
     * @param name -- Source name.
     * @param entry -- Entry pointer Source belongs to.
     */
    Source(const std::string &name, Entry *entry)
      : name(name), entry(entry) { }
    /**
     * @brief Clone constructor.
     * @param name -- other Source to get the name from.
     * @param entry -- Entry pointer Source belongs to.
     */
    Source(const Source &other, Entry *entry)
      : name(other.name), entry(entry) { }

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

  /**
   * @brief Base exception definition for Atypes and Rtypes classes.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class TypeError: public std::runtime_error {
  public:
    /** @brief Constructor.
     *  @param message -- error message.
     */
    TypeError(const std::string &message)
      : std::runtime_error(message) { }
  };

  /**
   * @brief Exception to be returned from Rtypes in case of output type error.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class SinkTypeError: public TypeError {
  public:
    SinkTypeError(const Sink *s, const std::string &message); ///< Constructor.

    const Sink *sink; ///< Sink pointer.
  };

  /**
   * @brief Exception to be returned from Atypes in case of input type error.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class SourceTypeError: public TypeError {
  public:
    SourceTypeError(const Source *s, const std::string &message); ///< Constructor.

    const Source *source; ///< Source pointer.
  };

  /**
   * @brief Exception to be returned from Rets in case of calculation error.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class CalculationError: public std::runtime_error {
  public:
    CalculationError(const Entry *e, const std::string &message); ///< Constructor.

    const Entry *entry; ///< Entry pointer.
  };

  class OutputHandle;
  /**
   * @brief Source wrapper to make it user accessible from the Python.
   * @copydetails OutputHandle
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class InputHandle {
    friend class OutputHandle;
  public:
    /**
     * @brief Constructor.
     * @param s -- Source to access.
     */
    InputHandle(Source &source): m_source(&source) { }
    /**
     * @brief Clone constructor.
     * @param other -- other InputHandle instance to access its Source.
     */
    InputHandle(const InputHandle &other): InputHandle(*other.m_source) { }

    // /**
    // * @brief
    // * @param name
    // * @todo method is undefined.
    // */
    // static InputHandle invalid(const std::string name);

    void connect(const OutputHandle &out) const; ///< Connect the Source to the other transformation's Sink via its OutputHandle

    const std::string &name() const { return m_source->name; } ///< Get Source's name.

    const void *rawptr() const { return static_cast<const void*>(m_source); } ///< Return Source's pointer as void pointer.
    const size_t hash() const { return reinterpret_cast<size_t>(rawptr()); }  ///< Return a Source's hash value based on it's pointer address.
  protected:
    TransformationTypes::Source *m_source; ///< Pointer to the Source.
  };

  /**
   * @brief Sink wrapper to make it user accessible from the Python.
   *
   * InputHandle and OutputHandle classes give indirect access to Source and Sink instances
   * and enable users to connect them in a calculation chain.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class OutputHandle {
    friend class InputHandle;
  public:
    /**
     * @brief Constructor.
     * @param s -- Sink to access.
     */
    OutputHandle(Sink &sink): m_sink(&sink) { }
    /**
     * @brief Clone constructor.
     * @param other -- other OutputHandle instance to access its Sink.
     */
    OutputHandle(const OutputHandle &other): OutputHandle(*other.m_sink) { }
    // /**
    // * @brief
    // * @param name
    // * @todo method is undefined.
    // */
    //static OutputHandle invalid(const std::string name);

    const std::string &name() const { return m_sink->name; } ///< Get Source's name.

    bool check() const; ///< Check the Entry.
    void dump() const;  ///< Dump the Entry.

    const double *data() const;                                     ///< Return pointer to the Sink's data buffer. Evaluate the data if needed in advance.
    const double *view() const { return m_sink->data->x.data(); }   ///< Return pointer to the Sink's data buffer without evaluation.
    const DataType &datatype() const { return m_sink->data->type; } ///< Return Sink's DataType.

    const void *rawptr() const { return static_cast<const void*>(m_sink); }  ///< Return Source's pointer as void pointer.
    const size_t hash() const { return reinterpret_cast<size_t>(rawptr()); } ///< Return a Source's hash value based on it's pointer address.

    bool depends(changeable x) const;  ///< Check that Sink depends on a changeable.
  private:
    TransformationTypes::Sink *m_sink; ///< Pointer to the Sink.
  };

  /**
   * @brief Connect the Source to the other transformation's Sink via its OutputHandle
   * @param out -- OutputHandle view to the Sink to connect to.
   */
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

  typedef boost::ptr_vector<Source> SourcesContainer;   ///< Container for Source pointers.
  typedef boost::ptr_vector<Sink> SinksContainer;       ///< Container for Sink pointers.
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
   * Entry will call the transformation function Entry::fun before returning
   * Data in case Entry is tainted or any of the Inputs is tainted.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Entry: public boost::noncopyable {
    Entry(const std::string &name, const Base *parent); ///< Constructor.
    Entry(const Entry &other, const Base *parent);      ///< Clone constructor.

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

    std::string name;                                   ///< Transformation name.
    SourcesContainer sources;                           ///< Transformation inputs (sources).
    SinksContainer sinks;                               ///< Transformation outputs (sinks).
    Function fun;                                       ///< The function that does actual calculation.
    std::vector<TypesFunction> typefuns;                ///< Vector of TypeFunction instances.
    taintflag tainted;                                  ///< taintflag shows whether the result is up to date.
    const Base *parent;                                 ///< Base class, containing the transformation Entry.
    int initializing;                                   ///< Initialization status. initializing>0 when Entry is being configured via Initializer.
    bool frozen;                                        ///< If Entry is frozen, it is not updated even if tainted.
    bool usable;                                        ///< Unused.

  private:
    template <typename InsT, typename OutsT>
    void initSourcesSinks(const InsT &inputs, const OutsT &outputs); ///< Initialize the clones for inputs and outputs.
  };
  typedef boost::ptr_vector<Entry> Container; ///< Container for Entry pointers.

  /**
   * @brief Return pointer to the Sink's data buffer. Evaluate the data if needed in advance.
   * @return pointer to the Sink's data buffer.
   */
  inline const double *OutputHandle::data() const {
    m_sink->entry->touch();
    return view();
  }

  /**
   * @brief Check that Sink depends on a changeable.
   * Simply checks that Entry depends on a changeable.
   * @param x -- changeable to test.
   * @return true if depends.
   */
  inline bool OutputHandle::depends(changeable x) const {
    return m_sink->entry->tainted.depends(x);
  }

  /**
   * @brief User-end Entry wrapper.
   *
   * This class gives an access to the transformation Entry.
   * It is inherited by TransformationDescriptor.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class Handle {
  public:
    Handle(): m_entry(nullptr) { }                            ///< Default constructor.
    Handle(Entry &entry) : m_entry(&entry) { }                ///< Constructor. @param entry -- an Entry instance to wrap.
    Handle(const Handle &other): Handle(*other.m_entry) { }   ///< Constructor. @param other -- Handle instance to get Entry to wrap.

    const std::string &name() const { return m_entry->name; } ///< Get entry name.
    std::vector<InputHandle> inputs() const;                  ///< Get vector of inputs.
    std::vector<OutputHandle> outputs() const;                ///< Get vector of outputs.

    /**
     * @brief Add named input.
     * @param name -- Source name.
     * @return InputHandle for the newly created Source.
     */
    InputHandle input(const std::string &name) {
      return m_entry->addSource(name);
    }

    InputHandle input(SingleOutput &output);                 ///< Create a new input and connect to the SingleOutput transformation.

    /**
     * @brief Add new named output.
     *
     * @param name -- new Sink's name.
     * @return OutputHandle for the new Sink.
     */
    OutputHandle output(const std::string &name) {
      return m_entry->addSink(name);
    }
    OutputHandle output(SingleOutput &output);               ///< Create a new output with a same name as SingleOutput's output.

    /**
     * @brief Return i-th Entry's Sink's data.
     *
     * The transformation function is evaluated if needed.
     *
     * @param i -- index of an output.
     * @return Data instance.
     */
    const Data<double> &operator[](int i) const { return m_entry->data(i); }

    /**
     * @brief Trigger an update of an Entry by simulating access to the i-th data.
     *
     * @param i -- Entry's Sink's index.
     */
    void update(int i) const { (void)m_entry->data(i); }
    void updateTypes() { m_entry->updateTypes(); }          ///< Call Entry::evaluateTypes(). @copydoc Entry::evaluateTypes()

    void unfreeze() { m_entry->frozen = false; }            ///< @copybrief Entry::unfreeze().

    void taint() { m_entry->tainted.taint(); }              ///< Taint the Entry's taintflag. The outputs will be evaluated upon request.
    taintflag tainted() { return m_entry->tainted; }        ///< Return the Entry's taintflag status.

    bool check() const { return m_entry->check(); }         ///< Call Entry::check(). @copydoc Entry::check()
    void dump() const { m_entry->dump(0); }                 ///< Call Entry::dump(). @copydoc Entry::dump()
    void dumpObj() const;                                   ///< Print Entry's Sink and Source instances and their connection status.
  protected:
    Entry *m_entry;                                         ///< Wrapped Entry pointer.
  };

  /**
   * @brief User-end wrapper for the Entry class that gives user an access to the actual Entry.
   *
   * The class is used for the dependency tree plotting via graphviz module.
   *
   * @author Maxim Gonchar
   * @date 12.2017
   */
  class OpenHandle : public Handle {
  public:
      OpenHandle(const Handle& other) : Handle(other){}; ///< Constructor. @param other -- Handle instance.
      Entry* getEntry() { return m_entry; }              ///< Get the Entry pointer.
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

    static void passAll(Atypes args, Rtypes rets);     ///< Assigns shape of each input to corresponding output.

    template <size_t Arg, size_t Ret = Arg>
    static void pass(Atypes args, Rtypes rets);        ///< Assigns shape of Arg-th input to Ret-th output.

    static void ifSame(Atypes args, Rtypes rets);      ///< Checks that all inputs are of the same type (shape and content description).
    static void ifSameShape(Atypes args, Rtypes rets); ///< Checks that all inputs are of the same shape.

    template <size_t Arg>
    static void ifHist(Atypes args, Rtypes rets);      ///< Checks if Arg-th input is a histogram (DataKind=Histogram).

    template <size_t Arg>
    static void ifPoints(Atypes args, Rtypes rets);    ///< Checks if Arg-th input is an array (DataKind=Points).

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

  /**
   * @brief Accessor gives an access to the Base's Entry instances by wrapping them into Handle.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class Accessor {
  public:
    Accessor() { };                                       ///< Default constructor.
    Accessor(Base &parent): m_parent(&parent) { }         ///< Constructor. @param parent -- Base instance to access its Entry instances.
    Handle operator[](int idx) const;                     ///< Get a Handle for the i-th Entry.
    Handle operator[](const std::string &name) const;     ///< Get a Handle for the Entry by name.
    size_t size() const;                                  ///< Get number of Entry instances.
  private:
    Base *m_parent;                                       ///< Pointer to the Base that keeps Entry instances.
  };

  template <typename T>
    class Initializer;

  /**
   * @brief Base transformation class handling.
   *
   * Base class does the bookkeeping for the transformations and defines the GNAObject transformation handling.
   *
   * Base class defines an object containing several transformation Entry instances.
   *
   * Each Entry defines an elementary transformation that will be updated in case any of Entry's inputs is updated.
   * Base enables the user to organize a more complex transformation each part of which depends on its own inputs
   * and thus may be updated independently. Entry instances within Base class may share internal data directly
   * (not via Sink-Source connections).
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class Base: public boost::noncopyable {
    template <typename T>
    friend class ::Transformation;
    template <typename T>
    friend class Initializer;
    friend class TransformationDescriptor;
    friend class Accessor;
    friend class ::GNAObject;
  public:
    Base(const Base &other);                                             ///< Clone constructor.
    Base &operator=(const Base &other);                                  ///< Clone assignment.
  protected:
    Base(): t_(*this) { }                                                ///< Default constructor.
    /**
     * @brief Constructor that limits the maximal number of allowed Entry instances.
     * @param maxentries -- maximal number of Entry instances the Base may keep.
     */
    Base(size_t maxentries): Base() {
      m_maxEntries = maxentries;
    }

    // Not implemented!
    // void connect(Source &source, Base *sinkobj, Sink &sink);

    /**
     * @brief Get Entry by index.
     * @param idx -- index of an Entry to return.
     * @return Entry.
     */
    Entry &getEntry(size_t idx) {
      return m_entries[idx];
    }
    Entry &getEntry(const std::string &name);                            ///< Get an Entry by name.

    template <typename T>
    Initializer<T> transformation_(T *obj, const std::string &name) {
      return Initializer<T>(obj, name);
    }

    Accessor t_;                                                         ///< An Accessor to Base's Entry instances via Handle.
  private:
    size_t addEntry(Entry *e);                                           ///< Add new Entry.
    boost::ptr_vector<Entry> m_entries;                                  ///< Vector of Entry pointers. Calls destructors when deleted.
    boost::optional<size_t> m_maxEntries;                                ///< Maximum number of allowed entries.
    void copyEntries(const Base &other);                                 ///< Clone entries from the other Base.
  };

  /**
   * @brief Get a Handle for the i-th Entry.
   * @param idx -- index of the Entry.
   * @return Handle for the Entry.
   */
  inline Handle Accessor::operator[](int idx) const {
    return Handle(m_parent->getEntry(idx));
  }

  /**
   * @brief Get a Handle for the Entry by name.
   * @param name -- Entry's name.
   * @return Handle for the Entry.
   */
  inline Handle Accessor::operator[](const std::string &name) const {
    TR_DPRINTF("accessing %s on %p\n", name.c_str(), (void*)m_parent);
    return Handle(m_parent->getEntry(name));
  }

  inline size_t Accessor::size() const {
    return m_parent->m_entries.size();
  }

  /**
   * @brief Transformation Entry initializer (CRTP).
   *
   * See
   * https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
   * for the CRTP description.
   *
   * Initializer is used in the transformation Base class via Base::transformation_() method
   * to add and configure new Entry instance. Each method of the Initializer returns this
   * thus allowing chain method call.
   *
   * Initializer enables user to add inputs, outputs, type functions and the trasnformation function.
   *
   * The typical usage is the following (from Identity transformation):
   * ```cpp
   * transformation_(this, "identity")
   *   .input("source")
   *   .output("target")
   *   .types(Atypes::pass<0,0>)
   *   .func([](Args args, Rets rets){ rets[0].x = args[0].x; })
   *   ;
   * ```
   * Initializer is usually used in the transformation's constructor and the scope is limited with
   * this constructor.
   *
   * @tparam T -- the actual transformation class being initialized.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template <typename T>
  class Initializer {
  public:
    /**
     * @brief Function, that does the actual calculation (reference to a member function).
     * @copydoc Function
     *
     * First argument is the actual transformation classe's `this` allowing to use member functions.
     *
     * @param args -- container with transformation inputs (Args).
     * @param rets -- container with transformation outputs (Args).
     */
    typedef std::function<void(T*,
                               TransformationTypes::Args,
                               TransformationTypes::Rets)> MemFunction;
    /**
     * @brief Function, that does the input types checking and output types derivation (reference to a member function).
     * @copydoc TypesFunction
     *
     * First argument is the actual transformation classe's `this` allowing to use member functions.
     *
     * @param atypes -- container with transformation inputs' types (Atypes).
     * @param rtypes -- container with transformation outputs' types (Rtypes).
     */
    typedef std::function<void(T*,
                               TransformationTypes::Atypes,
                               TransformationTypes::Rtypes)> MemTypesFunction;

    /**
     * @brief Constructor.
     *
     * Constructor increments Entry::initializing flag value thus indicating that Entry
     * is currently being configured via Initializer.
     *
     * @param obj -- Transformation pointer to manage. Used to get Base pointer for Entry.
     * @param name -- new Entry name.
     */
    Initializer(Transformation<T> *obj, const std::string &name)
      : m_entry(new Entry(name, obj->baseobj())), m_obj(obj),
        m_nosubscribe(false)
    {
      m_entry->initializing++;
    }
    /**
     * @brief Destructor.
     *
     * Destructor decrements the Entry::initializing flag value.
     * If Entry::initializing==0 then Initializer::add() method is called
     * which adds Entry instance to the Base.
     */
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
    /**
     * @brief Add the Entry to the Base.
     *
     * The method:
     *   - checks that the number of Entry instances in the Base does not
     *     exceed the maximal number of allowed entries.
     *   - passes Atypes::passAll() as TypeFunction if no TypeFunction pointers are provided.
     *   - subscribes the Entry to the Base's taint flag unless Initializer::m_nosubscribe is set.
     *   - adds the Entry to the Base.
     *   - adds MemFunction and MemTypesFunction pointers to the Transformation Initializer::m_obj.
     *
     * @note while Function and TypeFunction instances are keept within Entry
     * instance, MemFunction and MemTypesFunction instances are managed via
     * Transformation instance (Initializer::m_obj).
     */
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

    /**
     * @brief Add a named input.
     *
     * Adds a new Source to the Entry.
     *
     * @param name -- input name.
     * @return `*this`.
     */
    Initializer<T> input(const std::string &name) {
      m_entry->addSource(name);
      return *this;
    }

    /**
     * @brief Add a named output.
     *
     * Adds a new Sink to the Entry.
     *
     * @param name -- output name.
     * @return `*this`.
     */
    Initializer<T> output(const std::string &name) {
      m_entry->addSink(name);
      return *this;
    }

    /**
     * @brief Set the Entry::fun Function.
     * @param fun -- the Function that defines the transformation.
     * @return `*this`.
     */
    Initializer<T> func(Function func) {
      m_mfunc = nullptr;
      m_entry->fun = func;
      return *this;
    }

    /**
     * @brief Set the Entry::fun from a MemFunction.
     *
     * The method sets Entry::fun to the fun with first argument binded to `this` of the transformation.
     *
     * @param fun -- the MemFunction that defines the transformation.
     * @return `*this`.
     */
    Initializer<T> func(MemFunction func) {
      using namespace std::placeholders;
      m_mfunc = func;
      m_entry->fun = std::bind(func, m_obj->obj(), _1, _2);
      return *this;
    }

    /**
     * @brief Add new TypesFunction to the Entry.
     * @param func -- the TypesFunction to be added.
     * @return `*this`.
     */
    Initializer<T> types(TypesFunction func) {
      m_entry->typefuns.push_back(func);
      return *this;
    }

    /**
     * @brief Add new TypesFunction to the Entry based on the MemTypesFunction.
     *
     * The method makes new TypesFunction by binding the MemTypesFunction first argument `this`
     * of the transformation.
     *
     * @param func -- the MemTypesFunction to be added.
     * @return `*this`.
     */
    Initializer<T> types(MemTypesFunction func) {
      using namespace std::placeholders;
      m_mtfuncs.emplace_back(m_entry->typefuns.size(), func);
      m_entry->typefuns.push_back(std::bind(func, m_obj->obj(), _1, _2));
      return *this;
    }

    /**
     * @brief Force Entry::evaluateTypes() call.
     *
     * Entry::evaluateTypes() is usually called when outputs are connected to the inputs of other
     * transformations. This function should be used in case when it's known
     * that transformation has no inputs and its DataType may be derived immediately.
     */
    Initializer<T> finalize() {
      m_entry->evaluateTypes();
      return *this;
    }

    /**
     * @brief Add two TypeFunction pointers at once.
     * @tparam FuncA -- first function type (TypeFunction or MemTypesFunction).
     * @tparam FuncB -- second function type (TypeFunction or MemTypesFunction).
     * @note template parameters are usually determined automatically based on passed function types.
     * @param func1 -- first function to add.
     * @param func2 -- second function to add.
     */
    template <typename FuncA, typename FuncB>
    Initializer<T> types(FuncA func1, FuncB func2) {
      types(func1);
      types(func2);
      return *this;
    }

    /**
     * @brief Add three TypeFunction pointers at once.
     * @tparam FuncA -- first function type (TypeFunction or MemTypesFunction).
     * @tparam FuncB -- second function type (TypeFunction or MemTypesFunction).
     * @tparam FuncB -- third function type (TypeFunction or MemTypesFunction).
     * @note template parameters are usually determined automatically based on passed function types.
     * @param func1 -- first function to add.
     * @param func2 -- second function to add.
     * @param func3 -- third function to add.
     */
    template <typename FuncA, typename FuncB, typename FuncC>
    Initializer<T> types(FuncA func1, FuncB func2, FuncC func3) {
      types(func1);
      types(func2);
      types(func3);
      return *this;
    }

    /**
     * @brief Subscribe the Entry to track changeable's taintflag.
     *
     * Calling this method implies that Entry should not be subscribed automatically
     * to any taintflag. The user has to call Initializer::depends() explicitly for each
     * taintflag emitter.
     *
     * I.e. the Initializer::m_nosubscribe flag is set.
     *
     * @tparam Changeable -- the changeable type.
     * @param v -- changeable with ::subscribe() method.
     * @return `*this`
     */
    template <typename Changeable>
    Initializer<T> depends(Changeable v) {
      v.subscribe(m_entry->tainted);
      m_nosubscribe = true;
      return *this;
    }

    /**
     * @brief Subscribe the Entry to 1 or more changeable's taintflags.
     *
     * The function recursively calls itself to process all the arguments.
     *
     * @tparam Changeable -- the changeable type.
     * @tparam ... -- parameter pack.
     * @param v -- changeable with ::subscribe() method.
     * @param ... -- all other changeable instances.
     * @return `*this`
     */
    template <typename Changeable, typename... Rest>
    Initializer<T> depends(Changeable v, Rest... rest) {
      depends(v);
      return depends(rest...);
    }

    /**
     * @brief Disable automatic Entry subscription to the taintflag emissions.
     *
     * Sets the Initializer::m_nosubscribe flag and disables subscription to
     * the Base taintflag emission.
     *
     * @return `*this`
     */
    Initializer<T> dont_subscribe() {
      m_nosubscribe = true;
      return *this;
    }

  protected:
    Entry *m_entry;             ///< New Entry pointer.
    Transformation<T> *m_obj;   ///< The Transformation object managing MemFunction and MemTypesFunction instances. Has a reference to the Base.

    MemFunction m_mfunc;        ///< MemFunction pointer.
    std::vector<std::tuple<size_t, MemTypesFunction>> m_mtfuncs; ///< MemTypesFunction pointers.

    bool m_nosubscribe;         ///< Flag forbidding automatic subscription to Base taintflag emissions.
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

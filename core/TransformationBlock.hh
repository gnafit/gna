#pragma once

#include <list>
#include <tuple>

#include "TransformationEntry.hh"
#include "Atypes.hh"
#include "Rtypes.hh"
#include "Args.hh"
#include "Rets.hh"

template <typename Derived>
class TransformationBlock;

namespace TransformationTypes {
  /**
   * @brief TransformationBlock Entry initializer (CRTP).
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
     *
     * First argument is the actual transformation classe's `this` allowing to use member functions.
     *
     * @copydoc Function
     */
    typedef std::function<void(T*, Args, Rets)> MemFunction;
    /**
     * @brief Function, that does the input types checking and output types derivation (reference to a member function).
     *
     * First argument is the actual transformation classe's `this` allowing to use member functions.
     *
     * @copydoc TypesFunction
     */
    typedef std::function<void(T*, Atypes, Rtypes)> MemTypesFunction;

    /**
     * @brief Constructor.
     *
     * Constructor increments Entry::initializing flag value thus indicating that Entry
     * is currently being configured via Initializer.
     *
     * @param obj -- TransformationBlock pointer to manage. Used to get Base pointer for Entry.
     * @param name -- new Entry name.
     */
    Initializer(TransformationBlock<T> *obj, const std::string &name)
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
     *   - passes Atypes::passAll() as TypeFunction if no TypeFunction objects are provided.
     *   - subscribes the Entry to the Base's taint flag unless Initializer::m_nosubscribe is set.
     *   - adds the Entry to the Base.
     *   - adds MemFunction and MemTypesFunction objects to the TransformationBlock Initializer::m_obj.
     *
     * @note while Function and TypeFunction objects are kept within Entry
     * instance, MemFunction and MemTypesFunction instances are managed via
     * TransformationBlock instance (Initializer::m_obj).
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
     * @brief Add two TypeFunction objects at once.
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
     * @brief Add three TypeFunction objects at once.
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
    Entry *m_entry;                  ///< New Entry pointer.
    TransformationBlock<T> *m_obj;   ///< The TransformationBlock object managing MemFunction and MemTypesFunction objects. Has a reference to the Base.

    MemFunction m_mfunc;             ///< MemFunction object.
    std::vector<std::tuple<size_t, MemTypesFunction>> m_mtfuncs; ///< MemTypesFunction objects.

    bool m_nosubscribe;              ///< Flag forbidding automatic subscription to Base taintflag emissions.
  }; /* class Initializer */

}

/**
 * @brief Base class for the transformation definition.
 *
 * Each GNA transformation class is defined by deriving to base classes:
 *   - GNAObject or GNASingleObject (deriving from TransformationTypes::Base and ParametrizedTypes::Base).
 *   - TransformationBlock<class>.
 *
 * TransformationBlock class does the bookkeeping for MemFunction and MemTypesFunction. By defining CRTP
 * TransformationBlock::obj() method it facilitates the binding of the first
 * argument of MemFunction and MemTypesFunction objects to `this` of the transformation.
 *
 * @tparam Derived -- derived class type. See CRTP concept.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
template <typename Derived>
class TransformationBlock {
public:
  TransformationBlock() { }                                                   ///< Default constructor.
  /**
   * @brief Clone constructor.
   *
   * The constructor copies the list of MemFunction objects and rebinds them to `this`.
   *
   * @param other -- TransformationBlock to copy MemFunction and MemTypesFunction objects from.
   */
  TransformationBlock(const TransformationBlock<Derived> &other)
    : m_memFuncs(other.m_memFuncs), m_memTypesFuncs(other.m_memTypesFuncs)
  {
    rebindMemFunctions();
  }

  /**
   * @brief Clone assignment. Works the same was as clone constructor.
   * @copydoc TransformationBlock::TransformationBlock(const TransformationBlock<Derived>&)
   */
  TransformationBlock<Derived> &operator=(const TransformationBlock<Derived> &other) {
    m_memFuncs = other.m_memFuncs;
    m_memTypesFuncs = other.m_memTypeFuncs;
    rebindMemFunctions();
    return *this;
  }

private:
  friend class TransformationTypes::Initializer<Derived>;
  typedef typename TransformationTypes::Initializer<Derived> Initializer;
  typedef typename Initializer::MemFunction MemFunction;
  typedef typename Initializer::MemTypesFunction MemTypesFunction;

  /**
   * @brief Return `this` casted to the Derived type (CRTP).
   * @return `this`.
   */
  Derived *obj() { return static_cast<Derived*>(this); }

  /**
   * @copydoc TransformationBlock::obj()
   */
  const Derived *obj() const { return static_cast<const Derived*>(this); }

  /**
   * @brief Return `this` cast to the TransformationTypes::Base.
   * @return `this`.
   */
  TransformationTypes::Base *baseobj() {
    return static_cast<TransformationTypes::Base*>(obj());
  }

  /**
   * @copydoc TransformationBlock::baseobj()
   */
  const TransformationTypes::Base *baseobj() const {
    return static_cast<const TransformationTypes::Base*>(obj());
  }

  std::list<std::tuple<size_t, MemFunction>> m_memFuncs;                      ///< List with MemFunction objects arranged correspondingly to each Entry from Base.
  std::list<std::tuple<size_t, size_t, MemTypesFunction>> m_memTypesFuncs;    ///< List with MemTypesFunction objects arranged correspondingly to each Entry from Base.

  /**
   * @brief Add new MemFunction.
   * @param idx -- Entry index.
   * @param func -- the function.
   */
  void addMemFunction(size_t idx, MemFunction func) {
    m_memFuncs.emplace_back(idx, func);
  }

  /**
   * @brief Add new MemTypesFunction.
   * @param idx -- Entry index.
   * @param fidx -- function index (Each Entry may have several TypeFunction objects).
   * @param func -- the function.
   */
  void addMemTypesFunction(size_t idx, size_t fidx, MemTypesFunction func) {
    m_memTypesFuncs.emplace_back(idx, fidx, func);
  }

  /**
   * @brief Bind each of the MemFunction and MemTypesFunction objects to `this` of the transformation.
   * The method replaces the relevant Function and TypesFunction of the Entry by the binded MemFunction and MemTypesFunction.
   */
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
}; /* class TransformationBlock */


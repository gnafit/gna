#pragma once

#include "TransformationEntry.hh"
#include "TransformationFunctionArgs.hh"
#include "TypesFunctions.hh"

template <typename Derived>
class TransformationBind;

namespace TransformationTypes {
  /**
   * @brief TransformationBind Entry initializer (CRTP).
   *
   * See
   * https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
   * for the CRTP description.
   *
   * Initializer is used in the TransformationBind class via TransformationBind::transformation_()
   * method to add and configure new Entry instance. Each method of the Initializer returns this
   * thus allowing chain method call.
   *
   * Initializer enables user to add inputs, outputs, type functions and the trasnformation function.
   *
   * The typical usage is the following (from Identity transformation):
   * ```cpp
   * transformation_("identity")
   *   .input("source")
   *   .output("target")
   *   .types(TypesFunctions::pass<0,0>)
   *   .func([](FunctionArgs fargs){ fargs.rets[0].x = fargs.args[0].x; })
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
    typedef std::function<void(T*, FunctionArgs)> MemFunction;
    /**
     * @brief Function, that does the input types checking and output types derivation (reference to a member function).
     *
     * First argument is the actual transformation classe's `this` allowing to use member functions.
     *
     * @copydoc TypesFunction
     */
    typedef std::function<void(T*, TypesFunctionArgs fargs)> MemTypesFunction;

    /**
     * @brief Constructor.
     *
     * Constructor increments Entry::initializing flag value thus indicating that Entry
     * is currently being configured via Initializer.
     *
     * @param obj -- TransformationBind pointer to manage. Used to get Base pointer for Entry.
     * @param name -- new Entry name.
     */
    Initializer(TransformationBind<T> *obj, const std::string &name)
      : m_entry(new Entry(name, obj)), m_obj(obj),
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
     *   - passes TypesFunctions::passAll() as TypeFunction if no TypeFunction objects are provided.
     *   - subscribes the Entry to the Base's taint flag unless Initializer::m_nosubscribe is set.
     *   - adds the Entry to the Base.
     *   - adds MemFunction and MemTypesFunction objects to the TransformationBind Initializer::m_obj.
     *
     * @note while Function and TypeFunction objects are kept within Entry
     * instance, MemFunction and MemTypesFunction instances are managed via
     * TransformationBind instance (Initializer::m_obj).
     */
    void add() {
      if (m_obj->m_maxEntries &&
          m_obj->m_entries.size()+1 > m_obj->m_maxEntries) {
        throw std::runtime_error("too much transformations");
      }
      if (m_entry->typefuns.empty()) {
        m_entry->typefuns.push_back(TypesFunctions::passAll);
      }
      m_entry->initializing = 0;
      if (!m_nosubscribe) {
        m_obj->obj()->subscribe(m_entry->tainted);
      }
      size_t idx = m_obj->addEntry(m_entry);
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
     * @brief Set Entry label.
     * @param label -- Entry label.
     * @return `*this`.
     */
    Initializer<T> label(const std::string &label) {
      m_entry->label=label;
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
      m_entry->fun = std::bind(func, m_obj->obj(), _1);
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
      m_entry->typefuns.push_back(std::bind(func, m_obj->obj(), _1));
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
    TransformationBind<T> *m_obj;    ///< The TransformationBind object managing MemFunction and MemTypesFunction objects.

    MemFunction m_mfunc;             ///< MemFunction object.
    std::vector<std::tuple<size_t, MemTypesFunction>> m_mtfuncs; ///< MemTypesFunction objects.

    bool m_nosubscribe;              ///< Flag forbidding automatic subscription to Base taintflag emissions.
  }; /* class Initializer */
}

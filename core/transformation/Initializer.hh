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
   *   .func([](FunctionArgs& fargs){ fargs.rets[0].x = fargs.args[0].x; })
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
    using Function = FunctionT<double,double>;
    using TypesFunction = TypesFunctionT<double,double>;
    using StorageTypesFunction = StorageTypesFunctionT<double,double>;
    using FunctionArgs = FunctionArgsT<double,double>;
    using TypesFunctionArgs = TypesFunctionArgsT<double,double>;
    using StorageTypesFunctionArgs = StorageTypesFunctionArgsT<double,double>;
    /**
     * @brief Function, that does the actual calculation (reference to a member function).
     *
     * First argument is the actual transformation classe's `this` allowing to use member functions.
     *
     * @copydoc Function
     */
    using MemFunction = std::function<void (T *, FunctionArgs &)>;

    /**
     * @brief A container for the named MemFunction instances.
     */
    using MemFunctionMap = std::map<std::string, MemFunction>;

    /**
     * @brief Function, that does the input types checking and output types derivation (reference to a member function).
     *
     * First argument is the actual transformation classe's `this` allowing to use member functions.
     *
     * @copydoc TypesFunction
     */
    using MemTypesFunction = std::function<void (T *, TypesFunctionArgs &)>;

    /**
     * @brief A container for the named MemTypesFunction instances.
     *
     * Each pair is a number of the corresponding TypesFunction in Entry::typefuns and the MemTypesFunction.
     */
    using MemTypesFunctionMap = std::vector<std::tuple<size_t, MemTypesFunction>>;

    /**
     * @brief Function, that does the internal types derivation (reference to a member function).
     *
     * First argument is the actual transformation classe's `this` allowing to use member functions.
     *
     * @copydoc StorageTypesFunction
     */
    using MemStorageTypesFunction = std::function<void (T *, StorageTypesFunctionArgs &)>;

    /**
     * @brief A container for the named StorageMemTypesFunction instances.
     *
     * For each Function with name `name` there is a vector of pairs.
     * Each pair is a number of the corresponding StorageTypesFunction in Entry::functions[name] and the MemStorageTypesFunction.
     */
    using MemStorageTypesFunctionMap = std::map<std::string, std::vector<std::tuple<size_t, MemStorageTypesFunction>>>;

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
        this->add();
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
     *   - adds MemFunction, MemTypesFunction and StorageMemTypesFunction
     *     objects to the TransformationBind instance Initializer::m_obj.
     *
     * @note while Function, TypeFunction and StorageTypesFunction objects are kept within Entry
     * instance, MemFunction, MemTypesFunction and MemStorageTypesFunction instances are managed via
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
      for (const auto& kv: m_mfuncs) {
        m_obj->addMemFunction(idx, kv.first, kv.second);
      }
      for (const auto &f: m_mtfuncs) {
        m_obj->addMemTypesFunction(idx, std::get<0>(f), std::get<1>(f));
      }
      for (const auto &kv: m_mstfuncs) {
        for (const auto &f: kv.second){
          m_obj->addMemStorageTypesFunction(idx, kv.first, std::get<0>(f), std::get<1>(f));
        }
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
    Initializer<T> input(const std::string &name, bool inactive=false) {
      m_entry->addSource(name, inactive);
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
     * @brief Set the main Function.
     * @param fun -- the Function that defines the transformation.
     * @return `*this`.
     */
    Initializer<T> func(Function afunc) {
      this->func("main", afunc);
      return *this;
    }

    /**
     * @brief Set the named Function.
     *
     * The method adds a function Function to the Entry::functions storage by name.
     * If the name is main, the function is used as default Entry::fun.
     *
     * @param name -- a name of a function.
     * @param fun -- the Function that defines the transformation.
     * @exception std::runtime error if function with name `name` already exists.
     * @return `*this`.
     */
    Initializer<T> func(const std::string& name, Function afunc) {
      if( m_entry->functions.find(name)!=m_entry->functions.end() ){
        auto msg = fmt::format("mem function {0} already exists", name.data());
        throw std::runtime_error(msg);
      }
      m_entry->functions[name]={afunc, {}};

      if(name=="main"){
        m_entry->fun=afunc;
        m_entry->funcname="main";
      }
      return *this;
    }

    /**
     * @brief Switch to a particular function implementation.
     * @param name -- the function name.
     * @return `*this`.
     */
    Initializer<T> switchFunc(const std::string& name) {
      m_entry->switchFunction(name);
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
     * @brief Set the main function bound to a MemFunction.
     *
     * The method sets Entry::functions['main'] to the function with first argument bound to `this` of the transformation.
     *
     * @param fun -- the MemFunction that defines the transformation.
     * @return `*this`.
     */
    Initializer<T> func(MemFunction mfunc) {
      this->func("main", mfunc);
      return *this;
    }

    /**
     * @brief Set the named Function bound to a MemFunction.
     *
     * The method adds a function Function to the Entry::functions storage by name.
     * The added function is bound to a passed MemFunction and `this` of the transformation.
     * If the name is main, the function is used as default Entry::fun.
     *
     * @param name -- a name of a function.
     * @param fun -- the Function that defines the transformation.
     * @return `*this`.
     */
    Initializer<T> func(const std::string& name, MemFunction mfunc) {
      m_mfuncs[name]=mfunc;
      this->func(name, m_obj->template bind<>(mfunc));
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
      m_mtfuncs.emplace_back(m_entry->typefuns.size(), func);
      m_entry->typefuns.push_back(m_obj->template bind<>(func));
      return *this;
    }

    /**
     * @brief Add new StorageTypesFunction for the main function to initialize the storage.
     * @param func -- the TypesFunction to be added.
     * @return `*this`.
     */
    Initializer<T> storage(StorageTypesFunction func) {
      this->storage("main", func);
      return *this;
    }

    /**
     * @brief Add new StorageTypesFunction bound to the MemStorageTypesFunction.
     * @param func -- the TypesFunction to be added.
     * @return `*this`.
     */
    Initializer<T> storage(MemStorageTypesFunction func) {
      this->storage("main", func);
      return *this;
    }

    /**
     * @brief Add new StorageTypesFunction to a particular function to initialize the storage.
     * @param name -- function name to add the storage initializer.
     * @param func -- the TypesFunction to be added.
     * @exception runtime_error in case function is not found.
     * @return `*this`.
     */
    Initializer<T> storage(const std::string& name, StorageTypesFunction func) {
      auto& fd = m_entry->functions.at(name);
      fd.typefuns.emplace_back(func);
      return *this;
    }

    /**
     * @brief Add new StorageTypesFunction bound the MemStorageTypesFunction;
     * @param name -- function name to add the storage initializer.
     * @param func -- the TypesFunction to be added.
     * @exception runtime_error in case function is not found.
     * @return `*this`.
     */
    Initializer<T> storage(const std::string& name, MemStorageTypesFunction func) {
      auto& fd = m_entry->functions.at(name);
      m_mstfuncs[name].emplace_back(fd.typefuns.size(), func);
      fd.typefuns.push_back(m_obj->template bind<>(func));
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
      this->types(func1);
      this->types(func2);
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
      this->types(func1);
      this->types(func2);
      this->types(func3);
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
      this->depends(v);
      return this->depends(rest...);
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
    Entry *m_entry;                        ///< New Entry pointer.
    TransformationBind<T> *m_obj;          ///< The TransformationBind object managing MemFunction and MemTypesFunction objects.

    MemFunctionMap m_mfuncs;               ///< MemFunction objects.
    MemTypesFunctionMap m_mtfuncs;         ///< MemTypesFunction objects.
    MemStorageTypesFunctionMap m_mstfuncs; ///< MemStorageTypesFunction objects.

    bool m_nosubscribe;                    ///< Flag forbidding automatic subscription to Base taintflag emissions.
  }; /* class Initializer */
}
